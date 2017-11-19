

#ifndef INVIDX_HPP
#define INVIDX_HPP

#include "query.hpp"
#include "sdsl/config.hpp"
#include "sdsl/int_vector.hpp"
#include "block_postings_list.hpp"
#include "util.hpp"
#include "generic_rank.hpp"
#include "bm25.hpp"
#include "impact.hpp"

using namespace sdsl;

class score_heap : public std::priority_queue<doc_score, std::vector<doc_score>, std::greater<doc_score>> {
public:
  score_heap(const size_t k) {
    for (int i=0; i<k; i++) { push({(uint64_t)-1,0.0}); }
  }
  result toresult() {
    result res;
    // return the top-k results
    while (size()>0 && top().doc_id==(uint64_t)-1) { pop(); }
    res.list.resize(size());
    for (size_t i=0;i<res.list.size();i++) {
      auto min = top(); pop();
      res.list[res.list.size()-1-i] = min;
    }
    return res;
  }
};

template<class t_pl = block_postings_list<128>,
         class t_rank = generic_rank>
class idx_invfile {
public:
  using size_type = sdsl::int_vector<>::size_type;
  using plist_type = t_pl;
  using ranker_type = t_rank;
private:
  // determine lists
  struct plist_wrapper {
    typename plist_type::const_iterator cur;
    typename plist_type::const_iterator end;
    double list_max_score;
    double f_t;
    plist_wrapper() = default;
    plist_wrapper(plist_type& pl) {
      f_t = pl.size(); 
      cur = pl.begin();
      end = pl.end();
      list_max_score = pl.list_max_score();
    }
  };
private:
  std::vector<plist_type> m_postings_lists;
  std::unique_ptr<ranker_type> ranker;

public:
  idx_invfile() = default;
  double m_F;
  double m_conjunctive_max;

  // Search constructor 
  idx_invfile(std::string& postings_file, const double F) : m_F(F)
  {
    
    std:: ifstream ifs(postings_file);
    if (ifs.is_open() != true){
      std::cerr << "Could not open file: " <<  postings_file << std::endl;
      exit(EXIT_FAILURE);
    }
    size_t num_lists; 
    read_member(num_lists,ifs);
    m_postings_lists.resize(num_lists);
    for (size_t i=0;i<num_lists;i++) {
      m_postings_lists[i].load(ifs);
    }
  }

  auto serialize(std::ostream& out, 
                 sdsl::structure_tree_node* v=NULL, 
                 std::string name="") const -> size_type {

    structure_tree_node* child = structure_tree::add_child(v, name, 
                                 util::class_name(*this));
    size_type written_bytes = 0;
    size_t num_lists = m_postings_lists.size();
    written_bytes += sdsl::serialize(num_lists,out,child,"num postings lists");
    for (const auto& pl : m_postings_lists) {
      written_bytes += sdsl::serialize(pl,out,child,"postings list");
    }
    structure_tree::add_size(child, written_bytes);
    return written_bytes;
  }

  // Loads the ranker data
  void load(std::vector<uint64_t> doc_len, uint64_t terms, uint64_t num_docs,
            postings_form postings_type){
    if (postings_type == FREQUENCY) {
      ranker = std::unique_ptr<ranker_type>(new rank_bm25(doc_len, terms, num_docs));
    }
    else if (postings_type == QUANTIZED) {
      ranker = std::unique_ptr<ranker_type>(new rank_impact);
    }
  }

  // Finds the posting with the least number of items remaining other than
  // the current ID
  uint32_t
  find_shortest_list(std::vector<plist_wrapper*>& postings_lists,
                     uint32_t end,
                     const uint64_t id) 
  {
    uint32_t itr = 0;
    if (itr != end) {
      size_t smallest = std::numeric_limits<size_t>::max();
      auto smallest_itr = itr;
      while (itr != end) {
        if (postings_lists[itr]->cur.remaining() < smallest && postings_lists[itr]->cur.docid() != id) {
          smallest = postings_lists[itr]->cur.remaining();
          smallest_itr = itr;
        }
        ++itr;
      }
      return smallest_itr;
    }
    return end;
  }

  void sort_list_by_id(std::vector<plist_wrapper*>& plists) {
    // delete if necessary
    auto del_itr = plists.begin();
    while (del_itr != plists.end()) {
      if ((*del_itr)->cur == (*del_itr)->end) {
        del_itr = plists.erase(del_itr);
      } else {
        del_itr++;
      }
    }
    // sort
    auto id_sort = [](const plist_wrapper* a,const plist_wrapper* b) {
      return a->cur.docid() < b->cur.docid();
    };
    std::sort(plists.begin(),plists.end(),id_sort);
  }

  // WAND-Forwarding: Forwards smallest list to provided ID
  void forward_lists(std::vector<plist_wrapper*>& postings_lists,
       uint32_t& pivot_list,
       const uint64_t id) {

    auto smallest_itr = find_shortest_list(postings_lists,pivot_list,id);
    // advance the smallest list to the new id
    postings_lists[smallest_itr]->cur.skip_to_id(id);

    if (postings_lists[smallest_itr]->cur == postings_lists[smallest_itr]->end) {
      // list is finished! reorder list by id
      sort_list_by_id(postings_lists);
      return;
    }

    // bubble it down!
    auto next = smallest_itr + 1;
    uint32_t list_end = postings_lists.size();
    while (next != list_end && 
           postings_lists[smallest_itr]->cur.docid() > postings_lists[next]->cur.docid()) {
      std::swap(postings_lists[smallest_itr],postings_lists[next]);
      smallest_itr = next;
      next++;
    }
  }

  // BMW-Forwarding: Forwards beyond current block config
  void forward_lists_bmw(std::vector<plist_wrapper*>& postings_lists,
                uint32_t& pivot_list, const uint64_t docid) {

    // Find the shortest list
    auto smallest_iter = find_shortest_list(postings_lists, pivot_list+1, docid);
    // Determine the next ID which might need to be evaluated
    uint32_t list_end = postings_lists.size();
    uint32_t iter = 0;
    auto end = pivot_list + 1;
    uint64_t candidate_id = std::numeric_limits<uint64_t>::max();

    // 'shallow' forwarding - a block-max array look-up
    while (iter != end) {
      // The last ID in the block [without needed to actually skip to it]
      uint64_t bid = postings_lists[iter]->cur.block_containing_id(docid);
      uint64_t block_candidate = postings_lists[iter]->cur.block_rep(bid) + 1;
      candidate_id = std::min(candidate_id, block_candidate);
      ++iter;
    }
    // If the pivot was not in the last list, we must also consider the
    // smallest DocID from the other remaining lists. Skipping this step
    // will result in loss of safe-to-k results.
    if (end != list_end) {
      candidate_id = std::min(candidate_id, postings_lists[end]->cur.docid());
    }

    // Corner case check
    if (candidate_id < docid)
      candidate_id = docid + 1;
   
    // Advance the smallest list to our new candidate
    postings_lists[smallest_iter]->cur.skip_to_id(candidate_id);
    
    // If the smallest list is finished, reorder the lists
    if (postings_lists[smallest_iter]->cur == postings_lists[smallest_iter]->end) {
      sort_list_by_id(postings_lists);
      return;
    }

    // Bubble it down.
    auto next = smallest_iter + 1;
    while (next != list_end && 
          postings_lists[smallest_iter]->cur.docid() > postings_lists[next]->cur.docid()) {
      std::swap(postings_lists[smallest_iter], postings_lists[next]);
      smallest_iter = next;
      ++next;
    }
  }

  // Block-Max specific candidate test. Tests that the current pivot's block-max
  // scores still exceed the heap threshold. Returns the block-max score sum and
  // a boolean (whether we should indeed score, or not)
  const std::pair<bool, double> 
  potential_candidate(std::vector<plist_wrapper*>& postings_lists,
                      uint32_t& pivot_list, const double threshold,
                      const uint64_t doc_id){

    uint32_t iter = 0;
    double block_max_score = postings_lists[pivot_list]->cur.block_max(); // pivot blockmax

    // Lists preceding pivot list block max scores
    while (iter != pivot_list) {
      uint64_t bid = postings_lists[iter]->cur.block_containing_id(doc_id);
      block_max_score += postings_lists[iter]->cur.block_max(bid);
      ++iter;
    }

    // block-max test
    if (block_max_score > threshold) {
      return {true,block_max_score};
    }
    return {false,block_max_score};
  }

  // Returns a pivot document and its candidate (UB estimated) score.
  // Conjunctive pivot selection, can be used by BMW and Wand algos
  void
  determine_candidate(std::vector<plist_wrapper*>& postings_lists,
                      /*output*/ uint32_t& itr, double& score) {
    // Return the doc in the last list since it's furtherest along (and
    // the only doc that may contain ALL terms). Also return our
    // pre-computed sum of all UB scores (was computed upon recieving query).
    itr = postings_lists.size() - 1;
    score = m_conjunctive_max;
  }

  // Returns a pivot document and its candidate (UB estimated) score.
  // For disjunctive processing, can be used by BMW and Wand algos.
  void
  determine_candidate(std::vector<plist_wrapper*>& postings_lists, double threshold,
                      /*output*/ uint32_t& itr, double& score) {

    threshold = threshold * m_F; //Theta push
    score = 0;
    itr = 0;
    auto end = postings_lists.size();
    while(itr != end) {
      score += postings_lists[itr]->list_max_score;
      if(score > threshold) {
        // forward to last list equal to pivot
        auto pivot_id = postings_lists[itr]->cur.docid();
        auto next = itr+1;
        while(next != end && postings_lists[next]->cur.docid() == pivot_id) {
          itr = next;
          score += postings_lists[itr]->list_max_score;
          ++next;
        }
        return;
      }
      ++itr;
    }
  }

  // Evaluates the pivot document
  double evaluate_pivot(std::vector<plist_wrapper*>& postings_lists,
                        score_heap& heap,
                        double potential_score,
                        const double threshold,
                        const size_t k) {

    auto doc_id = postings_lists[0]->cur.docid(); //Pivot ID
    double doc_score = 0;
    double W_d = ranker->doc_length(doc_id);
    auto itr = postings_lists.begin();
    auto end = postings_lists.end();
    // Iterate postings 
    while (itr != end) {
      // Score the document if
      if ((*itr)->cur.docid() == doc_id) {
        double contrib = ranker->calculate_docscore((*itr)->cur.freq(),
                                                   (*itr)->f_t,
                                                   W_d);
        doc_score += contrib;
        potential_score += contrib;
        potential_score -= (*itr)->list_max_score; //Incremental refinement
        ++((*itr)->cur); // move to next larger doc_id
        // Check the refined potential max score 
        if (potential_score < threshold) {
          // Doc can no longer make the heap. Forward relevant lists.
          itr++;
          while (itr != end && (*itr)->cur != (*itr)->end 
                            && (*itr)->cur.docid() == doc_id) {
            ++((*itr)->cur);
            itr++;
          }
          break;
        }
      } 
      else {
        break;
      }
      itr++;
    }
    // add if it is in the top-k
    if (heap.top().score < doc_score) {
      heap.pop();
      heap.push({doc_id,doc_score});
    }
    // resort
    sort_list_by_id(postings_lists);
    return heap.top().score;
  }

  // Block-Max pivot evaluation
  double evaluate_pivot_bmw(std::vector<plist_wrapper*>& postings_lists,
                        score_heap& heap,
                        double potential_score,
                        const double threshold,
                        const size_t k) {

    uint64_t doc_id = postings_lists[0]->cur.docid(); // pivot
    double doc_score = 0;
    double W_d = ranker->doc_length(doc_id);
    auto itr = postings_lists.begin();
    auto end = postings_lists.end();
    
    // Iterate PLs
    while (itr != end) {
      // If we have the pivot, contribute the score
      if ((*itr)->cur.docid() == doc_id) {
        double contrib = ranker->calculate_docscore((*itr)->cur.freq(),
                                                   (*itr)->f_t,
                                                   W_d);
        doc_score += contrib;
        potential_score += contrib;
        // Differs from WAND version as we use BM scores for estimation
        uint64_t bid = (*itr)->cur.block_containing_id(doc_id);
        potential_score -= (*itr)->cur.block_max(bid);
        ++((*itr)->cur); // move to next larger doc_id
        // Doc cannot make heap, but we need to forward lists anyway 
        if (potential_score < threshold) {
          // move the other equal ones ahead still! 
          itr++;
          while (itr != end && (*itr)->cur != (*itr)->end 
                            && (*itr)->cur.docid() == doc_id) {
            ++((*itr)->cur);
            itr++;
          }
          break;
        }  
      } 
      else {
        break;
      }
      itr++;
    }
    // add if it is in the top-k
    if (heap.top().score < doc_score) {
      heap.pop();
      heap.push({doc_id,doc_score});
    }
    // resort
    sort_list_by_id(postings_lists);
    return heap.top().score;
  }


  // Wand Disjunctive Algorithm
  result process_wand_disjunctive(std::vector<plist_wrapper*>& postings_lists,
                                  const size_t k) {
    score_heap heap(k);

    // init list processing 
    double threshold = heap.top().score;

    // Initial Sort, get the pivot and its potential score
    sort_list_by_id(postings_lists);
    uint32_t pivot_list;
    double potential_score;
    determine_candidate(postings_lists, threshold, pivot_list, potential_score);

    // While our pivot doc is not the end of the PL
    while (pivot_list != postings_lists.size()) {
      // If the first posting ID is that of the pivot, evaluate!
      if (postings_lists[0]->cur.docid() == postings_lists[pivot_list]->cur.docid()) {
          threshold = evaluate_pivot(postings_lists,
                                     heap,
                                     potential_score,
                                     threshold,
                                     k);
      }
      // We must forward the lists before the puvot up to our pivot doc  
      else {
        forward_lists(postings_lists,pivot_list,postings_lists[pivot_list]->cur.docid());
      }
      // Grsb the next pivot and its potential score
      determine_candidate(postings_lists, threshold, pivot_list, potential_score);
    }

    return heap.toresult();
  }

  // Wand Conjunctive Algorithm
  result process_wand_conjunctive(std::vector<plist_wrapper*>& postings_lists,
                                  const size_t k) {
    score_heap heap(k);

    // init list processing 
    double threshold = heap.top().score;
    // Initial Sort, get the pivot and its potential score
    sort_list_by_id(postings_lists);
    size_t initial = postings_lists.size();
    uint32_t pivot_list;
    double potential_score;
    determine_candidate(postings_lists, pivot_list, potential_score);

    // While our pivot doc is not the end of the PL and we have not exhausted
    // any of our PL's
    while (pivot_list != postings_lists.size() &&
                postings_lists.size() == initial) {
      // If the first posting ID is that of the pivot, evaluate!
      if (postings_lists[0]->cur.docid() == postings_lists[pivot_list]->cur.docid()) {
          threshold = evaluate_pivot(postings_lists,
                                     heap,
                                     potential_score,
                                     threshold,
                                     k);
      }
      // We must forward the lists before the pivot up to our pivot doc  
      else {
        forward_lists(postings_lists,pivot_list,postings_lists[pivot_list]->cur.docid());
      }
      // Grsb the next pivot and its potential score
      determine_candidate(postings_lists, pivot_list, potential_score);
    }

    return heap.toresult();
  }

  // BlockMax Wand Disjunctive
  result process_bmw_disjunctive(std::vector<plist_wrapper*>& postings_lists,
                            const size_t k){   
    score_heap heap(k);

    // init list processing , grab first pivot and potential score
    double threshold = heap.top().score;
    sort_list_by_id(postings_lists);
    uint32_t pivot_list;
    double potential_score;
    determine_candidate(postings_lists, threshold, pivot_list, potential_score);

    // While we have got documents left to evaluate
    while (pivot_list != postings_lists.size()) {
      uint64_t candidate_id = postings_lists[pivot_list]->cur.docid();
      // Second level candidate check
      auto candidate_and_score = potential_candidate(postings_lists, pivot_list,
                                          threshold, candidate_id);
      auto candidate = std::get<0>(candidate_and_score);
      auto potential_score = std::get<1>(candidate_and_score);
      // If the document is still a candidate from BM scores
      if (candidate) {
        // If lists are aligned for pivot, score the doc
        if (postings_lists[0]->cur.docid() == candidate_id) {
          threshold = evaluate_pivot_bmw(postings_lists, heap,
                                     potential_score, threshold, k);
        }
        // Need to forward list before the pivot 
        else {
          forward_lists(postings_lists,pivot_list,candidate_id);
        }
      }
      // Use the knowledge that current block-max config can not yield a
      // solution, and skip to the next possible fruitful configuration
      else {
        forward_lists_bmw(postings_lists,pivot_list,candidate_id); 
      }
      // Grab a new pivot and keep going!
      determine_candidate(postings_lists, threshold, pivot_list, potential_score);
    }

    return heap.toresult();
  }

  // BlockMax Wand Conjunctive
  result process_bmw_conjunctive(std::vector<plist_wrapper*>& postings_lists,
                                const size_t k){   
    score_heap heap(k);

    // init list processing , grab first pivot and potential score
    double threshold = heap.top().score;
    sort_list_by_id(postings_lists);
    uint32_t pivot_list;
    double potential_score;
    determine_candidate(postings_lists, pivot_list, potential_score);
    size_t initial = postings_lists.size();

    // While we have got documents left to evaluate
    while (pivot_list != postings_lists.size() &&
                         postings_lists.size() == initial) {
      uint64_t candidate_id = postings_lists[pivot_list]->cur.docid();
      // Second level candidate check
      auto candidate_and_score = potential_candidate(postings_lists, pivot_list,
                                          threshold, candidate_id);
      auto candidate = std::get<0>(candidate_and_score);
      auto potential_score = std::get<1>(candidate_and_score);
      // If the document is still a candidate from BM scores
      if (candidate) {
        // If lists are aligned for pivot, score the doc
        if (postings_lists[0]->cur.docid() == candidate_id) {
          threshold = evaluate_pivot_bmw(postings_lists, heap,
                                     potential_score, threshold, k);
        }
        // Need to forward list before the pivot 
        else {
          forward_lists(postings_lists,pivot_list,candidate_id);
        }
      }
      // Use the knowledge that current block-max config can not yield a
      // solution, and skip to the next possible fruitful configuration
      else {
        forward_lists_bmw(postings_lists,pivot_list,candidate_id); 
      }
      // Grab a new pivot and keep going!
      determine_candidate(postings_lists, pivot_list, potential_score);
    }

    return heap.toresult();
  }


  result search(const std::vector<query_token>& qry, const size_t k,
                const index_form t_index_type,
                const query_traversal t_index_traversal) {

    m_conjunctive_max = 0.0f; // Reset for new query
    std::vector<plist_wrapper> pl_data(qry.size());
    std::vector<plist_wrapper*> postings_lists;
    size_t j=0;
    for (const auto& qry_token : qry) {
      pl_data[j] = plist_wrapper(m_postings_lists[qry_token.token_id]);
      postings_lists.emplace_back(&(pl_data[j]));
      m_conjunctive_max += pl_data[j].list_max_score;
      ++j;
    }

    // Select and run query
    if (t_index_type == BMW) {
      if (t_index_traversal == OR)
        return process_bmw_disjunctive(postings_lists,k);
      else if (t_index_traversal == AND)
        return process_bmw_conjunctive(postings_lists,k);
    }

    else if (t_index_type == WAND) {
      if (t_index_traversal == OR)
        return process_wand_disjunctive(postings_lists,k);
      else if (t_index_traversal == AND)
        return process_wand_conjunctive(postings_lists,k);
    }
    
    else {
      std::cerr << "Invalid run-type selected. Must be wand or bmw."
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

};

// Search
template<class t_pl,class t_rank>
void construct(idx_invfile<t_pl,t_rank> &idx,
               std::string& postings_file, 
                const double F)
{
    using namespace sdsl;
    cout << "construct(idx_invfile)"<< endl;
    idx = idx_invfile<t_pl,t_rank>(postings_file, F);
    cout << "Done" << endl;
}
#endif

