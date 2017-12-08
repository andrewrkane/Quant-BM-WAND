#include <iostream>

#include "ant_param_block.h"
#include "search_engine.h"
#include "search_engine_btree_leaf.h"
#include "btree_iterator.h"
#include "memory.h"

#include "include/generic_rank.hpp"
#include "include/bm25.hpp"
#include "include/impact.hpp"

#include "sdsl/int_vector_buffer.hpp"
#include "include/block_postings_list.hpp"
#include "include/util.hpp"

const static size_t INIT_SZ = 4096; 
const static size_t INDRI_OFFSET = 2; // Indri offsets terms 0 and 1 as special


int main(int argc, char **argv)
{
	ANT_ANT_param_block params(argc-3, argv);
	long last_param = params.parse();

	if (last_param == argc)
	{
		std::cout << "USAGE: " << argv[0];
		std::cout << " [ATIRE options] <NONE|RANDOM|-|orderfilename> <collection folder> <index_type>\n"
              << " index type can be `BMW` or `WAND` or `SPWAND`" << std::endl;
		return EXIT_FAILURE;
	}
	using clock = std::chrono::high_resolution_clock;

  std::string order_file = argv[argc-3];
	std::string collection_folder = argv[argc-2];
  std::string s_index_type = argv[argc-1];
	create_directory(collection_folder);
	std::string dict_file = collection_folder + "/dict.txt";
	std::string doc_names_file = collection_folder + "/doc_names.txt";
	std::string postings_file = collection_folder + "/WANDbl_postings.idx";
	std::string global_info_file = collection_folder + "/global.txt";
	std::string doclen_tfile = collection_folder + "/doc_lens.txt";
  std::string index_type_file = collection_folder + "/index_info.txt";

	std::ofstream doclen_out(doclen_tfile);

	auto build_start = clock::now();

  // Select the index format - BMW or WAND or SPWAND?
  index_form index_format;
  if (s_index_type == STRING_BMW) {
    index_format = BMW;
  }
  else if (s_index_type == STRING_WAND) {
    index_format = WAND;
  }
  else if (s_index_type == STRING_SPWAND) {
    index_format = SPWAND;
  }
  else {
    std::cerr << "Incorrect index type specified. Exiting." << std::endl;
    return EXIT_FAILURE;
  }

  // For reference (later, for a user), write out which index type this is
  std::ofstream index_file_output(index_type_file);
  index_file_output << s_index_type << std::endl;

  // load stuff
  ANT_memory memory;
  ANT_search_engine search_engine(&memory);
  search_engine.open(params.index_filename);

  // Keep track of term ordering
  unordered_map<string, uint64_t> map;

  std::vector<uint64_t> doclen_vector;

  std::cout << "Writing global info to " << global_info_file << "."
            << std::endl;
  std::vector<std::string> document_names;

  // dump global info; num documents in collection, num of all terms
  std::ofstream of_globalinfo(global_info_file);
  of_globalinfo << search_engine.document_count() << " "
                << search_engine.term_count() << std::endl;

  // AK: reordering list
  vector<int> reorder;
  vector<int> reorderInv;
  int doc_count = search_engine.document_count();
  int order_count = doc_count;
  reorder.reserve(doc_count);
  reorderInv.reserve(doc_count);
  for (uint32_t i = 0; i < doc_count; i++) {
    doclen_vector.emplace_back(-1); // now index is reorderd id rather than original id.
  }
  if (order_file == "NONE") {
    for (uint32_t i = 0; i < doc_count; i++) {
      reorder.emplace_back(i); reorderInv.emplace_back(i);
    }
  } else if (order_file == "RANDOM") {
    // AK: random order
    std::cout << "Computing random order." << std::endl;
    for (uint32_t i = 0; i < doc_count; i++) { reorder.emplace_back(i); }
    srand(time(NULL));
    for (uint32_t i = 0; i < doc_count; i++) {
      int r = rand() % doc_count;
      std::swap(reorder[i],reorder[r]);
    }
    for (uint32_t i = 0; i < doc_count; i++) { reorderInv[reorder[i]] = i; }
  } else {
    for (uint32_t i = 0; i < doc_count; i++) {
      reorder.emplace_back(-1); reorderInv.emplace_back(-1);
    }
    // AK: input order
    std::cout << "Computing reordering from " << order_file << "." << std::endl;
    // load order file
    shared_ptr<istream> order_file_input;
    if (order_file == "-")
      order_file_input.reset(&cin, [](...){});
    else
      order_file_input.reset(new ifstream(order_file));
    //std::ifstream order_file_input(order_file);
    vector<pair<string, uint32_t>> order_names;
    order_names.reserve(doc_count);
    for (int i = 0;; i++) {
      string line; getline(*order_file_input, line);
      if (order_file_input->eof()) {
        if (i <= 0) { std::cerr << "Error: Order file is empty." << std::endl; exit(1); }
        if (i != doc_count) { std::cerr<<"Found only "<<i<<" documents in order files."<<std::endl; } break;
      }
      if (i>doc_count) { std::cerr<<"Too many values in order file (max="<<doc_count<<") "<<order_file<<std::endl; return -1; }
      order_names.emplace_back(line, i);
    }
    order_count = order_names.size();
    // setup original order
    vector<pair<string, uint32_t>> orig_names;
    orig_names.reserve(doc_count);
    long long start = search_engine.get_variable(ATIRE_DOCUMENT_FILE_START);
    long long end = search_engine.get_variable(ATIRE_DOCUMENT_FILE_END);
    unsigned long bsize = end - start;
    char *buffer = (char *)malloc(bsize);
    auto filenames = search_engine.get_document_filenames(buffer, &bsize);
    for (int i = 0; i < doc_count; i++) {
      orig_names.emplace_back(filenames[i], i);
    }
    // sorted join
    std::sort(std::begin(order_names), std::end(order_names));
    std::sort(std::begin(orig_names), std::end(orig_names));
    int i = 0;
    for (int j = 0; i < order_count && j < doc_count; j++) {
      if (order_names[i].first < orig_names[j].first) { std::cerr << "Names in order file do not agree with index " << order_file << " " << order_names[i].first << " " << orig_names[i].first << std::endl; return -1; }
      if (order_names[i].first > orig_names[j].first) { continue; } // skip missing names in order file
      reorder[orig_names[j].second] = order_names[i].second;
      reorderInv[order_names[i].second] = orig_names[j].second;
      i++;
    }
    // cleanup
    free(buffer);
    // debugging
    //std::cerr << "doc_count=" << doc_count << std::endl; for (uint32_t i = 0; i < 10; i++) { std::cerr << reorder[i] << " " << reorderInv[reorder[i]] << std::endl; }
  }

  // write the lengths and names
  {
    std::cout << "Writing document lengths to " << doclen_tfile << "."
      << std::endl;
    std::cout << "Writing document names to " << doc_names_file << "." 
      << std::endl;
    std::ofstream of_doc_names(doc_names_file);
    
    long long start = search_engine.get_variable(ATIRE_DOCUMENT_FILE_START);
    long long end = search_engine.get_variable(ATIRE_DOCUMENT_FILE_END);
    unsigned long bsize = end - start;
    char *buffer = (char *)malloc(bsize);
    auto filenames = search_engine.get_document_filenames(buffer, &bsize);

    uint64_t uniq_terms = search_engine.get_unique_term_count();
    // Shift all IDs from ATIRE by 2 so \0 and \1 are free.
    uniq_terms += 2; 

    
    double mean_length;
    auto lengths = search_engine.get_document_lengths(&mean_length);
    {
      for (long long i = 0; i < search_engine.document_count(); i++)
      {
        // AK: reorder
        if (reorderInv[i] < 0) { continue; } // skip missing names in order file
        doclen_out << lengths[reorderInv[i]] << std::endl;
        of_doc_names << filenames[reorderInv[i]] << std::endl;
        doclen_vector[reorderInv[i]]=lengths[i];
      }
    }

    free(buffer);
  }
  // write dictionary
  {
    std::cout << "Writing dictionary to " << dict_file << "." << std::endl;
    std::ofstream of_dict(dict_file);

    ANT_search_engine_btree_leaf leaf;
    ANT_btree_iterator iter(&search_engine);

    size_t j = 2;
    for (char *term = iter.first(NULL); term != NULL; term = iter.next()) {
      iter.get_postings_details(&leaf);
      of_dict << term << " " << j << " "
        << leaf.local_document_frequency << " "
        << leaf.local_collection_frequency << " "
        << "\n";
      map.emplace(strdup(term), j);
      j++;
    }
  }

  // ranker is a unique_ptr to the ranker type
  std::unique_ptr<generic_rank> ranker;
  // Use quant ranker
  
  if (search_engine.quantized()) {
    ranker = std::unique_ptr<generic_rank>(new rank_impact); 
    std::cerr << "You provided a pre-quantized ATIRE index, so I am building" 
              << " a quantized index." << std::endl;
    index_file_output << STRING_QUANT << std::endl; // keep track of index type
  }
  else {
    ranker = std::unique_ptr<generic_rank>(new rank_bm25(doclen_vector, search_engine.term_count()));
    std::cerr << "You provided a frequency ATIRE index, so I am building" 
              << " a frequency index." << std::endl;
    index_file_output << STRING_FREQ << std::endl; // keep track of index type

  }

  // write inverted files
  {
    // output 1st,10th,50th,100th,500th,1000th and 2000th highest impacts for bootstrap thresholds
    uint64_t list_thresholds_id=2;
    ofstream list_thresholds_out_1(collection_folder+"/list_thresholds_1.txt");
    ofstream list_thresholds_out_10(collection_folder+"/list_thresholds_10.txt");
    ofstream list_thresholds_out_50(collection_folder+"/list_thresholds_50.txt");
    ofstream list_thresholds_out_100(collection_folder+"/list_thresholds_100.txt");
    ofstream list_thresholds_out_500(collection_folder+"/list_thresholds_500.txt");
    ofstream list_thresholds_out_1000(collection_folder+"/list_thresholds_1000.txt");
    ofstream list_thresholds_out_2000(collection_folder+"/list_thresholds_2000.txt");

    using plist_type = block_postings_list<128>;
    vector<plist_type> m_postings_lists;
    vector<vector<pair<uint64_t, uint64_t>>> temp_postings_lists;
    uint64_t a = 0, b = 0;
    uint64_t n_terms = search_engine.get_unique_term_count() + INDRI_OFFSET; // + 2 to skip 0 and 1
 
    vector<pair<uint64_t, uint64_t>> post; 
    post.reserve(INIT_SZ);
    // AK: SPLITLISTS
    vector<pair<uint64_t, uint64_t>> post2;
    post2.reserve(INIT_SZ);

    // Open the files
    filebuf post_file;
    post_file.open(postings_file, std::ios::out);
    ostream ofs(&post_file);

    std::cerr << "Generating postings lists ..." << std::endl;

    m_postings_lists.resize(n_terms);


    ANT_search_engine_btree_leaf leaf;
    ANT_btree_iterator iter(&search_engine);
    ANT_impact_header impact_header;
    ANT_compression_factory factory;

    ANT_compressable_integer *raw;
    long long impact_header_size = ANT_impact_header::NUM_OF_QUANTUMS * sizeof(ANT_compressable_integer) * 3;
    ANT_compressable_integer *impact_header_buffer = (ANT_compressable_integer *)malloc(impact_header_size);
    auto postings_list_size = search_engine.get_postings_buffer_length();
    auto raw_list_size = sizeof(*raw) * (search_engine.document_count() + ANT_COMPRESSION_FACTORY_END_PADDING);
    unsigned char *postings_list = (unsigned char *)malloc((size_t)postings_list_size);
    raw = (ANT_compressable_integer *)malloc((size_t)raw_list_size);
    uint64_t term_count = 0;

    size_t num_lists = n_terms;
    cout << "Writing " << num_lists << " postings lists." << endl;
    sdsl::serialize(num_lists, ofs);

    // take the 0 and 1 terms with dummies
    sdsl::serialize(block_postings_list<128>(), ofs);
    // AK: SPLITLISTS
    if (index_format == SPWAND) sdsl::serialize(block_postings_list<128>(), ofs);
    sdsl::serialize(block_postings_list<128>(), ofs);
    // AK: SPLITLISTS
    if (index_format == SPWAND) sdsl::serialize(block_postings_list<128>(), ofs);

     for (char *term = iter.first(NULL); term != NULL; term_count++, term = iter.next())
    {
	// don't capture ~ terms, they are specific to ATIRE
      if (*term == '~')
        break;

      iter.get_postings_details(&leaf);
      postings_list = search_engine.get_postings(&leaf, postings_list);

      auto the_quantum_count = ANT_impact_header::get_quantum_count(postings_list);
      auto beginning_of_the_postings = ANT_impact_header::get_beginning_of_the_postings(postings_list);
      factory.decompress(impact_header_buffer, postings_list + ANT_impact_header::INFO_SIZE, the_quantum_count * 3);

      if (term_count % 100000 == 0) {
      /* if (true) { */
        std::cout << term << " @ " << leaf.postings_position_on_disk << " (cf:" << leaf.local_collection_frequency << ", df:" << leaf.local_document_frequency << ", q:" << the_quantum_count << ")" << std::endl;
		fflush(stdout);
      }

      long long docid, max_docid, sum;
      ANT_compressable_integer *impact_header = (ANT_compressable_integer *)impact_header_buffer;
      ANT_compressable_integer *current, *end;

      max_docid = sum = 0;
      ANT_compressable_integer *impact_value_ptr = impact_header;
      ANT_compressable_integer *doc_count_ptr = impact_header + the_quantum_count;
      ANT_compressable_integer *impact_offset_start = impact_header + the_quantum_count * 2;
      ANT_compressable_integer *impact_offset_ptr = impact_offset_start;

      post.clear();
      post.reserve(leaf.local_document_frequency);
      // AK: SPLITLISTS
      if (index_format == SPWAND) {
        post2.clear();
        post2.reserve(leaf.local_document_frequency);
      }

      // AK: SPLITLISTS
      bool bSecondList = false;

      while (doc_count_ptr < impact_offset_start) {
        // AK: SPLITLISTS >10k and use 10% of list size up to change of impact/quantum
        static int splitListsMinSize = 10000;
        if (index_format == SPWAND && !bSecondList
            && leaf.local_document_frequency>splitListsMinSize
            && post.size()>((double)leaf.local_document_frequency)*0.10) {
          bSecondList = true;
        }

        factory.decompress(raw, postings_list + beginning_of_the_postings + *impact_offset_ptr, *doc_count_ptr);
        docid = -1;
        current = raw;
        end = raw + *doc_count_ptr;
        while (current < end) {
          docid += *current++;
          // AK: SPLITLISTS
          if (index_format == SPWAND && bSecondList) {
            post2.emplace_back(reorder[docid], *impact_value_ptr);
          } else {
            post.emplace_back(reorder[docid], *impact_value_ptr);
          }
        }
        impact_value_ptr++;
        impact_offset_ptr++;
        doc_count_ptr++;
      }

#define GETPOST(x) (post.size()>=x ? post[x-1].second : post2[x-post.size()-1].second)
      // AK: output 1st,10th,50th,100th,500th,1000th and 2000th highest impact if exists (to bootstrap threshold values)
      size_t s = post.size()+post2.size();
      if (s>=1) { list_thresholds_out_1<<list_thresholds_id<<" "<<GETPOST(1)<<endl;
        if (s>=10) { list_thresholds_out_10<<list_thresholds_id<<" "<<GETPOST(10)<<endl;
          if (s>=50) { list_thresholds_out_50<<list_thresholds_id<<" "<<GETPOST(50)<<endl;
            if (s>=100) { list_thresholds_out_100<<list_thresholds_id<<" "<<GETPOST(100)<<endl;
              if (s>=500) { list_thresholds_out_500<<list_thresholds_id<<" "<<GETPOST(500)<<endl;
                if (s>=1000) { list_thresholds_out_1000<<list_thresholds_id<<" "<<GETPOST(1000)<<endl;
                  if (s>=2000) { list_thresholds_out_2000<<list_thresholds_id<<" "<<GETPOST(2000)<<endl;
                  }
                }
              }
            }
          }
        }
      }
      list_thresholds_id++;

      // AK: reordering could cause prune list to be empty
      if (post.size() > 0) {
        // The above will result in sorted by impact first, so re-sort by docid
        std::sort(std::begin(post), std::end(post));
        plist_type pl(ranker, post, index_format);
        sdsl::serialize(pl, ofs);
      } else {
        sdsl::serialize(block_postings_list<128>(), ofs);
      }
      // AK: SPLITLISTS
      if (index_format == SPWAND) {
        if (post2.size() > 0) {
          // The above will result in sorted by impact first, so re-sort by docid
          std::sort(std::begin(post2), std::end(post2));
          plist_type pl(ranker, post2, index_format);
          sdsl::serialize(pl, ofs);
          // debugging
          //if (post.size()+post2.size()>100000) std::cerr<<" sp("<<post.size()<<","<<post2.size()<<")";
        } else {
          sdsl::serialize(block_postings_list<128>(), ofs);
        }
      }

    }
    //close output files
    post_file.close();
  }

	auto build_stop = clock::now();
	auto build_time_sec = std::chrono::duration_cast<std::chrono::seconds>(build_stop-build_start);
	std::cout << "Index built in " << build_time_sec.count() << " seconds." << std::endl;

	return EXIT_SUCCESS;
}
