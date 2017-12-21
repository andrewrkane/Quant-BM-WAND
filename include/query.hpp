#ifndef QUERY_HPP
#define QUERY_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>

#include "util.hpp"


struct doc_score {
	uint64_t doc_id;
	double score;
  bool operator>(const doc_score& rhs) const {
  	if(score == rhs.score)
    	return doc_id > rhs.doc_id;
      return score > rhs.score;
    }
  doc_score() {};
  doc_score(uint64_t did, double s) : doc_id(did) , score(s) {};
};

struct result {
  std::vector<doc_score> list;
  uint64_t qry_id = 0;
  uint64_t wt_search_space = 0;
  uint64_t wt_nodes = 0;
  uint64_t postings_evaluated = 0; // Sum calls to "evaluate_pivot"
  uint64_t postings_total = 0; // Sum of lengths of postings lists from query
  uint64_t docs_fully_evaluated = 0;
  uint64_t docs_added_to_heap = 0; 
  double final_threshold = 0; // Final top-k heap threshold
};

struct query_token{
    uint64_t token_id;
    std::string token_str;
    uint64_t f_qt;
	query_token(const uint64_t id,
              const std::string str,
              uint64_t f) : token_id(id), token_str(str), 
              f_qt(f) 
    {
    }
};

// AK: added start_heap to capture thresholds defined from postings lists at indexing time
using query_t = std::tuple<uint64_t,std::vector<query_token>,double>; // query_id, query_tokens, start_heap

#include <strstream>
std::string itos( int n ) {
  std::stringstream ss;
  ss << n;
  return ss.str();
}

struct query_parser {
    query_parser() = delete;
    using mapping_t = std::tuple<std::unordered_map<std::string,uint64_t>,
                     std::unordered_map<uint64_t,std::string>,
                     std::unordered_map<uint64_t,uint64_t>
                     >;

    static void
         load_dictionary(const std::string& collection_dir, mapping_t& mapping, const size_t list_threshold_k)
    {
        std::unordered_map<uint64_t,uint64_t>& start_heap_mapping = std::get<2>(mapping);
        if (list_threshold_k > 0) {
            // AK: load also from list_thresholds_#.txt
            auto list_thresholds_file = collection_dir + "/list_thresholds_" + itos(list_threshold_k) + ".txt";
            std::cerr<<"loading "<<list_thresholds_file<<std::endl;
            std::ifstream ifs(list_thresholds_file);
            if(!ifs.is_open()) {
                std::cerr << "cannot load list_thresholds file.";
                exit(EXIT_FAILURE);
            }
            std::string impact_mapping;
            while( std::getline(ifs,impact_mapping) ) {
                auto sep_pos = impact_mapping.find(' ');
                uint64_t id = std::stoull(impact_mapping.substr(0,sep_pos));
                uint64_t list_thresholds = std::stoull(impact_mapping.substr(sep_pos+1));
                start_heap_mapping[id] = list_thresholds;
            }
            std::cerr<<"done loading "<<list_thresholds_file<<std::endl;
        }
        std::unordered_map<std::string,uint64_t>& id_mapping = std::get<0>(mapping);
        std::unordered_map<uint64_t,std::string>& reverse_id_mapping = std::get<1>(mapping);
        {
            auto dict_file = collection_dir + "/" + DICT_FILENAME;
            std::cerr<<"loading "<<dict_file<<std::endl;
            std::ifstream dfs(dict_file);
            if(!dfs.is_open()) {
                std::cerr << "cannot load dictionary file.";
                exit(EXIT_FAILURE);
            }
            std::string term_mapping;
            while( std::getline(dfs,term_mapping) ) {
                auto sep_pos = term_mapping.find(' ');
                auto term = term_mapping.substr(0,sep_pos);
                auto idstr = term_mapping.substr(sep_pos+1);
                uint64_t id = std::stoull(idstr);
                id_mapping[term] = id;
                reverse_id_mapping[id] = term;
            }
            std::cerr<<"done loading "<<dict_file<<std::endl;
        }
    }

    static std::tuple<bool,uint64_t,std::vector<uint64_t>> 
        map_to_ids(const std::unordered_map<std::string,uint64_t>& id_mapping,
                   std::string query_str,bool only_complete,bool integers)
    {
        auto id_sep_pos = query_str.find(';');
        auto qryid_str = query_str.substr(0,id_sep_pos);
        auto qry_id = std::stoull(qryid_str);
        auto qry_content = query_str.substr(id_sep_pos+1);

        std::vector<uint64_t> ids;
        std::istringstream qry_content_stream(qry_content);
        for(std::string qry_token; std::getline(qry_content_stream,qry_token,' ');) {
            if(integers) {
                uint64_t id = std::stoull(qry_token);
                ids.push_back(id);
            } else {
                auto id_itr = id_mapping.find(qry_token);
                if(id_itr != id_mapping.end()) {
                    ids.push_back(id_itr->second);
                } else {
                    std::cerr << "ERROR: could not find '" 
                              << qry_token << "' in the dictionary." 
                              << std::endl;
                    if(only_complete) {
                        return std::make_tuple(false,qry_id,ids);
                    }
                }
            }
        }
        return std::make_tuple(true,qry_id,ids);
    }

    static std::pair<bool,query_t> parse_query(const mapping_t& mapping,
                const std::string& query_str,
                bool only_complete = false,bool integers = false)
    {
        double start_heap = 0.0;

        const auto& id_mapping = std::get<0>(mapping);
        const auto& reverse_mapping = std::get<1>(mapping);
        const auto& start_heap_mapping = std::get<2>(mapping);

        auto mapped_qry = map_to_ids(id_mapping,query_str,only_complete,integers);

        bool parse_ok = std::get<0>(mapped_qry);
        auto qry_id = std::get<1>(mapped_qry);

        if(parse_ok) {
            std::unordered_map<uint64_t,uint64_t> qry_set;
            const auto& tids = std::get<2>(mapped_qry);
            for(const auto& tid : tids) {
                qry_set[tid] += 1;
                // AK: start_heap using max top-k from lists
                if (start_heap_mapping.find(tid) != start_heap_mapping.end()) {
                  if (start_heap_mapping.at(tid)>start_heap) start_heap = start_heap_mapping.at(tid);
                }
            }
            std::vector<query_token> query_tokens;
            size_t index = 0;
            for(const auto& qry_tok : qry_set) {
                uint64_t term = qry_tok.first;
                auto rmitr = reverse_mapping.find(term);
                std::string term_str;
                if(rmitr != reverse_mapping.end()) {
                    term_str = rmitr->second;
                }
                query_tokens.emplace_back(term,term_str,qry_tok.second);
                ++index;
            }
            query_t q(qry_id,query_tokens,fmax(0.0,start_heap-1)); // AK: start_heap minus one so can find all in single list
            return {true,q};
        }

        // error
        query_t q;
        return {false,q};
    }

    static std::vector<query_t> parse_queries(const std::string& collection_dir,
                                              const std::string& query_file,
                                              const size_t list_threshold_k = 0,
                                              bool only_complete = false) {
        std::vector<query_t> queries;

        // AK: pass mapping as a variable, because returning as value copies these large data structures
        mapping_t mapping;

        /* load the mapping */
        load_dictionary(collection_dir, mapping, list_threshold_k);
        /* parse queries */
        std::ifstream qfs(query_file); 
        if(!qfs.is_open()) {
            std::cerr << "cannot load query file.";
            exit(EXIT_FAILURE);
        }

        std::string query_str;
        while( std::getline(qfs,query_str) ) {
            auto parsed_qry = parse_query(mapping,query_str);
            if(parsed_qry.first) {
                queries.emplace_back(parsed_qry.second);
            }
        }

        return queries;
    }
};

#endif
