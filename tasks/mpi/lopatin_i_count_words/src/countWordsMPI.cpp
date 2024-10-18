#include "mpi/lopatin_i_count_words/include/countWordsMPIHeader.hpp"

namespace lopatin_i_count_words_mpi {

int countWords(const std::string& str) {
  std::istringstream iss(str);
  return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
}

std::string generateLongString(int n) {
  std::string testData;
  std::string testString = "This is a long sentence for performance testing of the word count algorithm using MPI. ";
  for (int i = 0; i < n; i++) {
    testData += testString;
  }
  return testData;
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
  wordCount = 0;
  return true;
}

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool TestMPITaskSequential::run() {
  internal_order_test();
  wordCount = countWords(input_);
  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  return true;
}

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();

  int total_words = 0;

  if (world.rank() == 0) {
    input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
    std::istringstream iss(input_);
    std::string word;
    while (iss >> word) {
      words.push_back(word);
    }
    total_words = words.size();
  }

  boost::mpi::broadcast(world, total_words, 0);

  int localWordsCountTemp = total_words / world.size();
  int remainder = total_words % world.size();

  int start = world.rank() * localWordsCountTemp + std::min(world.rank(), remainder);
  int end = start + localWordsCountTemp + (world.rank() < remainder ? 1 : 0);

  localWordCount = 0;
  if (world.rank() == 0) {
    localWordCount = end - start;

    for (int i = 1; i < world.size(); ++i) {
      int chunk_start = i * localWordsCountTemp + std::min(i, remainder);
      int chunk_end = chunk_start + localWordsCountTemp + (i < remainder ? 1 : 0);
      int chunk_size = chunk_end - chunk_start;
      world.send(i, 0, chunk_size);
    }
  } else {
    world.recv(0, 0, localWordCount);
  }

  wordCount = 0;
  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  return (world.rank() == 0) ? (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1) : true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();

  boost::mpi::reduce(world, localWordCount, wordCount, std::plus<>(), 0);

  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = wordCount;
  }
  return true;
}

}  // namespace lopatin_i_count_words_mpi