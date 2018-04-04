

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
  score_heap(const size_t k, double start_heap) {
    for (int i=0; i<k; i++) { push({(uint64_t)-1,start_heap}); }
  }
  result toresult() {
    // AK: debugging
    std::cerr<<" [heap.top="<<top().score<<"]";

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
    size_t list_size;
    plist_wrapper* pair; // AK: SPLITLISTS
    uint64_t pair_max_score; // AK: SPLITLISTS
    plist_wrapper() = default;
    plist_wrapper(plist_type& pl) {
      f_t = list_size = pl.size();
      cur = pl.begin();
      end = pl.end();
      list_max_score = pl.list_max_score();
      // AK: SPLITLISTS
      pair=NULL;
      pair_max_score=0;
    }
  };
private:
  std::vector<plist_type> m_postings_lists;
  std::unique_ptr<ranker_type> ranker;

public:
  double m_F;
  double m_conjunctive_max;

  // Search constructor
  idx_invfile(std::string& postings_file, const double F, index_form& t_index_type)
  {
    cout << "construct(idx_invfile)"<< endl;
    m_F = F;
    std:: ifstream ifs(postings_file);
    if (ifs.is_open() != true){
      std::cerr << "Could not open file: " <<  postings_file << std::endl;
      exit(EXIT_FAILURE);
    }
    size_t num_lists; 
    read_member(num_lists,ifs);
    // AK: SPLITLISTS
    if (t_index_type == SPWAND || t_index_type == SPBMW) num_lists*=2;
    m_postings_lists.resize(num_lists);
    for (size_t i=0;i<num_lists;i++) {
      m_postings_lists[i].load(ifs);
    }
    cout << "Done" << endl;
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

  // Finds the smallest postings list before the specified ID
  size_t
  find_shortest_list(std::vector<plist_wrapper*>& postings_lists,
                     size_t end,
                     const uint64_t id) 
  {
    size_t itr = 0;
    if (itr != end) {
      size_t smallest = postings_lists[itr]->list_size;
      auto smallest_itr = itr;
      itr++;
      while (itr != end) {
        size_t rem = postings_lists[itr]->list_size;
        if (rem < smallest && postings_lists[itr]->cur.docid() != id) {
          smallest = rem;
          smallest_itr = itr;
        }
        ++itr;
      }
      return smallest_itr;
    }
    return end;
  }

  static int id_sort(const plist_wrapper* a,const plist_wrapper* b) {
    return a->cur.docid() < b->cur.docid();
  };

  void sort_list_by_id(std::vector<plist_wrapper*>& plists) {
    // delete if necessary
    for (size_t di=0; di<plists.size();) {
      if (plists[di]->cur == plists[di]->end) {
        // remove by bubble up and pop
        for (size_t si=di; si+1<plists.size(); ++si) {
          std::swap(plists[si],plists[si+1]);
        }
        plists.pop_back();
      } else {
        ++di;
      }
    }
    // sort
    std::sort(plists.begin(),plists.end(),id_sort);
  }

  // WAND-Forwarding: Forwards smallest list to provided ID
  void forward_lists(std::vector<plist_wrapper*>& postings_lists,
       size_t& pivot_list,
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
    size_t list_end = postings_lists.size();
    while (next != list_end && 
           postings_lists[smallest_itr]->cur.docid() > postings_lists[next]->cur.docid()) {
      std::swap(postings_lists[smallest_itr],postings_lists[next]);
      smallest_itr = next;
      ++next;
    }
  }

  // BMW-Forwarding: Forwards beyond current block config
  void forward_lists_bmw(std::vector<plist_wrapper*>& postings_lists,
                size_t& pivot_list, const uint64_t docid) {

    // Find the shortest list
    auto smallest_iter = find_shortest_list(postings_lists, pivot_list+1, docid);
    // Determine the next ID which might need to be evaluated
    size_t list_end = postings_lists.size();
    size_t iter = 0;
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
  const bool
  potential_candidate(std::vector<plist_wrapper*>& postings_lists,
                      size_t& pivot_list, const double threshold,
                      const uint64_t doc_id, /*output*/ double& block_max_score){

    // AK: SPLITLISTS
    if (false && !SPLITNOCHECK && SPLITLISTS) { // AK: remove this for now since always slower
      size_t iter = 0;
      block_max_score = 0;
      // Lists preceding pivot list block max scores
      while (iter <= pivot_list) {
        uint64_t bid = postings_lists[iter]->cur.block_containing_id(doc_id);
        // AK: if both iter and pair to be counted then count only when pair before iter
        if (postings_lists[iter]->pair->cur != postings_lists[iter]->pair->end
            && postings_lists[iter]->pair->cur.docid() <= doc_id
            && postings_lists[iter]->pair->cur.docid() <= postings_lists[iter]->cur.docid()) {
          uint64_t bid_pair = postings_lists[iter]->pair->cur.block_containing_id(doc_id);
          block_max_score += max(postings_lists[iter]->cur.block_max(bid),
                                 postings_lists[iter]->pair->cur.block_max(bid_pair));
        } else {
          block_max_score += postings_lists[iter]->cur.block_max(bid);
        }
        ++iter;
      }
    }
    // AK: !SPLITLISTS
    else {
      size_t iter = 0;
      block_max_score = postings_lists[pivot_list]->cur.block_max(); // pivot blockmax

      // Lists preceding pivot list block max scores
      while (iter != pivot_list) {
        uint64_t bid = postings_lists[iter]->cur.block_containing_id(doc_id);
        block_max_score += postings_lists[iter]->cur.block_max(bid);
        ++iter;
      }
    }

    // block-max test
    if (block_max_score > threshold) {
      return true;
    }
    return false;
  }

  // Returns a pivot document and its candidate (UB estimated) score.
  // Conjunctive pivot selection, can be used by BMW and Wand algos
  void
  determine_candidate(std::vector<plist_wrapper*>& postings_lists,
                      /*output*/ size_t& itr, double& score) {
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
                      /*output*/ size_t& itr, double& score) {

    threshold = threshold * m_F; //Theta push
    score = 0;
    itr = 0;
    auto end = postings_lists.size();
    while(itr != end) {
      // AK: SPLITLISTS
      if (!SPLITNOCHECK && SPLITLISTS
          && postings_lists[itr]->pair->cur != postings_lists[itr]->pair->end
          && postings_lists[itr]->pair->cur.docid() <= postings_lists[itr]->cur.docid()) {
        score += postings_lists[itr]->pair_max_score;
      } else {
        score += postings_lists[itr]->list_max_score;
      }
      if(score > threshold) {
        // forward to last list equal to pivot
        auto pivot_id = postings_lists[itr]->cur.docid();
        auto next = itr+1;
        while(next != end && postings_lists[next]->cur.docid() == pivot_id) {
          itr = next;
          // AK: SPLITLISTS
          if (!SPLITNOCHECK && SPLITLISTS
              && postings_lists[itr]->pair->cur != postings_lists[itr]->pair->end
              && postings_lists[itr]->pair->cur.docid() <= postings_lists[itr]->cur.docid()) {
            score += postings_lists[itr]->pair_max_score;
          } else {
            score += postings_lists[itr]->list_max_score;
          }
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
                        const double threshold) {

    auto doc_id = postings_lists[0]->cur.docid(); //Pivot ID
    double doc_score = 0;
    double W_d = ranker->doc_length(doc_id);
    size_t itr = 0;
    size_t end = postings_lists.size();
    // Iterate postings 
    while (itr != end) {
      // Score the document if
      if (postings_lists[itr]->cur.docid() == doc_id) {
        double contrib = ranker->calculate_docscore(postings_lists[itr]->cur.freq(),
                                                   postings_lists[itr]->f_t,
                                                   W_d);
        doc_score += contrib;
        potential_score += contrib;
        potential_score -= postings_lists[itr]->list_max_score; //Incremental refinement
        ++(postings_lists[itr]->cur); // move to next larger doc_id
        // Check the refined potential max score 
        if (potential_score < threshold) {
          // Doc can no longer make the heap. Forward relevant lists.
          ++itr;
          while (itr != end && postings_lists[itr]->cur != postings_lists[itr]->end
                            && postings_lists[itr]->cur.docid() == doc_id) {
            ++(postings_lists[itr]->cur);
            ++itr;
          }
          break;
        }
      } 
      else {
        break;
      }
      ++itr;
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
                        const double threshold) {

    uint64_t doc_id = postings_lists[0]->cur.docid(); // pivot
    double doc_score = 0;
    double W_d = ranker->doc_length(doc_id);
    size_t itr = 0;
    size_t end = postings_lists.size();
    
    // Iterate PLs
    while (itr != end) {
      // If we have the pivot, contribute the score
      if (postings_lists[itr]->cur.docid() == doc_id) {
        double contrib = ranker->calculate_docscore(postings_lists[itr]->cur.freq(),
                                                   postings_lists[itr]->f_t,
                                                   W_d);
        doc_score += contrib;
        potential_score += contrib;
        // Differs from WAND version as we use BM scores for estimation
        uint64_t bid = postings_lists[itr]->cur.block_containing_id(doc_id);
        potential_score -= postings_lists[itr]->cur.block_max(bid);
        ++(postings_lists[itr]->cur); // move to next larger doc_id
        // Doc cannot make heap, but we need to forward lists anyway 
        if (potential_score < threshold) {
          // move the other equal ones ahead still! 
          ++itr;
          while (itr != end && postings_lists[itr]->cur != postings_lists[itr]->end
                            && postings_lists[itr]->cur.docid() == doc_id) {
            ++(postings_lists[itr]->cur);
            ++itr;
          }
          break;
        }  
      } 
      else {
        break;
      }
      ++itr;
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
  result process_wand_disjunctive(std::vector<plist_wrapper*>& postings_lists, score_heap& heap) {
    // init list processing
    double threshold = heap.top().score;

    // Initial Sort, get the pivot and its potential score
    sort_list_by_id(postings_lists);
    size_t pivot_list;
    double potential_score;
    determine_candidate(postings_lists, threshold, pivot_list, potential_score);

    // While our pivot doc is not the end of the PL
    while (pivot_list != postings_lists.size()) {
      // If the first posting ID is that of the pivot, evaluate!
      if (postings_lists[0]->cur.docid() == postings_lists[pivot_list]->cur.docid()) {
          threshold = evaluate_pivot(postings_lists, heap, potential_score, threshold);
      }
      // We must forward the lists before the pivot up to our pivot doc
      else {
        forward_lists(postings_lists,pivot_list,postings_lists[pivot_list]->cur.docid());
      }
      // Grsb the next pivot and its potential score
      determine_candidate(postings_lists, threshold, pivot_list, potential_score);
    }

    return heap.toresult();
  }

  // Wand Conjunctive Algorithm
  result process_wand_conjunctive(std::vector<plist_wrapper*>& postings_lists, score_heap& heap) {
    // init list processing
    double threshold = heap.top().score;

    // Initial Sort, get the pivot and its potential score
    sort_list_by_id(postings_lists);
    size_t initial = postings_lists.size();
    size_t pivot_list;
    double potential_score;
    determine_candidate(postings_lists, pivot_list, potential_score);

    // While our pivot doc is not the end of the PL and we have not exhausted
    // any of our PL's
    while (pivot_list != postings_lists.size() &&
                postings_lists.size() == initial) {
      // If the first posting ID is that of the pivot, evaluate!
      if (postings_lists[0]->cur.docid() == postings_lists[pivot_list]->cur.docid()) {
          threshold = evaluate_pivot(postings_lists, heap, potential_score, threshold);
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
  result process_bmw_disjunctive(std::vector<plist_wrapper*>& postings_lists, score_heap& heap){
    // init list processing , grab first pivot and potential score
    double threshold = heap.top().score;
    sort_list_by_id(postings_lists);
    size_t pivot_list;
    double potential_score;
    determine_candidate(postings_lists, threshold, pivot_list, potential_score);

    // While we have got documents left to evaluate
    while (pivot_list != postings_lists.size()) {
      uint64_t candidate_id = postings_lists[pivot_list]->cur.docid();
      // Second level candidate check
      auto candidate = potential_candidate(postings_lists, pivot_list,
                                          threshold, candidate_id, potential_score);
      // If the document is still a candidate from BM scores
      if (candidate) {
        // If lists are aligned for pivot, score the doc
        if (postings_lists[0]->cur.docid() == candidate_id) {
          threshold = evaluate_pivot_bmw(postings_lists, heap, potential_score, threshold);
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
  result process_bmw_conjunctive(std::vector<plist_wrapper*>& postings_lists, score_heap& heap){
    // init list processing , grab first pivot and potential score
    double threshold = heap.top().score;
    sort_list_by_id(postings_lists);
    size_t pivot_list;
    double potential_score;
    determine_candidate(postings_lists, pivot_list, potential_score);
    size_t initial = postings_lists.size();

    // While we have got documents left to evaluate
    while (pivot_list != postings_lists.size() &&
                         postings_lists.size() == initial) {
      uint64_t candidate_id = postings_lists[pivot_list]->cur.docid();
      // Second level candidate check
      auto candidate = potential_candidate(postings_lists, pivot_list,
                                          threshold, candidate_id, potential_score);
      // If the document is still a candidate from BM scores
      if (candidate) {
        // If lists are aligned for pivot, score the doc
        if (postings_lists[0]->cur.docid() == candidate_id) {
          threshold = evaluate_pivot_bmw(postings_lists, heap, potential_score, threshold);
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

  void usage() {
    std::cerr<<"Invalid run-type/index-traversal combination. Must be "
             <<(SPLITLISTS?"SPWAND(OR) or BMW(OR)":"WAND(AND|OR) or BMW(AND|OR)")
             <<"."<<std::endl;
  }


  result search(const std::vector<query_token>& qry, const size_t k,
                const double start_heap,
                const index_form t_index_type,
                const query_traversal t_index_traversal) {

    // validate parameters before loading index
    if ((SPLITLISTS && t_index_type == BMW)
        || (SPLITLISTS && t_index_type == WAND)
        || (!SPLITLISTS && t_index_type == SPBMW)
        || (!SPLITLISTS && t_index_type == SPWAND)) {
      usage();
      exit(EXIT_FAILURE);
    }

    m_conjunctive_max = 0.0f; // Reset for new query
    // AK: SPLITLISTS
    std::vector<plist_wrapper> pl_data(t_index_type == SPWAND || t_index_type == SPBMW ? 2*qry.size() : qry.size());
    std::vector<plist_wrapper*> postings_lists;
    size_t j=0;
    for (const auto& qry_token : qry) {
      m_conjunctive_max += pl_data[j].list_max_score;
      if (t_index_type == SPWAND || t_index_type == SPBMW) {
        // AK: SPLITLISTS are encoded in series, so 2x and 2x+1, and pair them together
        pl_data[j] = plist_wrapper(m_postings_lists[2*qry_token.token_id]);
        postings_lists.emplace_back(&(pl_data[j]));
        ++j;
        pl_data[j] = plist_wrapper(m_postings_lists[2*qry_token.token_id+1]);
        postings_lists.emplace_back(&(pl_data[j]));
        ++j;
        pl_data[j-1].pair = &pl_data[j-2];
        pl_data[j-2].pair = &pl_data[j-1];
        pl_data[j-1].pair_max_score = max(0, (int)(pl_data[j-1].list_max_score) - (int)(pl_data[j-1].pair->list_max_score));
        pl_data[j-2].pair_max_score = max(0, (int)(pl_data[j-2].list_max_score) - (int)(pl_data[j-2].pair->list_max_score));
      } else {
        pl_data[j] = plist_wrapper(m_postings_lists[qry_token.token_id]);
        postings_lists.emplace_back(&(pl_data[j]));
        ++j;
      }
    }
    // AK: debugging
    //std::cerr<<" [sizes";for (int i=0; i<pl_data.size(); ++i) { std:cerr<<" "<<pl_data[i].list_size; } std::cerr<<"]";

    score_heap heap(k, start_heap);

    // AK: debugging
    std::cerr<<" [start_heap="<<start_heap<<"]";

    // Select and run query
    if (!SPLITLISTS && t_index_type == BMW) {
      if (t_index_traversal == OR)
        return process_bmw_disjunctive(postings_lists,heap);
      else if (t_index_traversal == AND)
        return process_bmw_conjunctive(postings_lists,heap);
    }

    else if (!SPLITLISTS && t_index_type == WAND) {
      if (t_index_traversal == OR)
        return process_wand_disjunctive(postings_lists,heap);
      else if (t_index_traversal == AND)
        return process_wand_conjunctive(postings_lists,heap);
    }

    else if (SPLITLISTS && t_index_type == SPBMW) {
      // AK: SPLITLISTS - differences from WAND processing come from setting SPLITLISTS variable
      if (t_index_traversal == OR)
        return process_bmw_disjunctive(postings_lists,heap);
    }

    else if (SPLITLISTS && t_index_type == SPWAND) {
      // AK: SPLITLISTS - differences from WAND processing come from setting SPLITLISTS variable
      if (t_index_traversal == OR)
        return process_wand_disjunctive(postings_lists,heap);
    }

    usage();
    exit(EXIT_FAILURE);
  }

};

#endif

