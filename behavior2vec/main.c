//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>

#define MAX_STRING 1000
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define MAX_RW_LENGTH 1005
#define MAX_RW_LINE_LENGTH 200005

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
	long long cn;
	long long word;
	int *point;
	char *code, codelen;
};

char train_file[MAX_STRING], output_file_entity[MAX_STRING], output_file_action[MAX_STRING];
char output_file_action_time[MAX_STRING], output_file_entity_time[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab_entity = NULL, *vocab_action = NULL;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce_entity = 1, min_redue_action = 1;
int timestep_size = -1;    // number of timesteps
int *vocab_hash_entity, *vocab_hash_action;
long long vocab_max_size_entity = 10000, vocab_max_size_action = 10000;
long long vocab_size_entity = 0, vocab_size_action = 0, layer1_size = 0;
long long train_entity = 0, train_action = 0;
long long train_behavior = 0, behavior_count_actual = 0, iter = 5, file_size = 0, classes = 0;
long long min_time = INT64_MAX, max_time = INT64_MIN;   // upper bound and lower bound of time
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0_entity = NULL, *syn1_entity = NULL, *syn1neg_entity = NULL;
real *syn0_action = NULL, *syn1_action = NULL, *syn1neg_action = NULL;
real *expTable;
clock_t start;


real *w_entity, *w_action;    // projection matrix for entity and action

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table_entity, *table_action;

int* InitUnigramTable(struct vocab_word *vocab, long long vocab_size) {
	int a, i;
	double train_words_pow = 0;
	double d1, power = 0.75;
	int *table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
	i = 0;
	d1 = pow(vocab[i].cn, power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (double)table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1;
	}
	return table;
}

void UpdateTimeBound(long long time) {
  max_time = time > max_time ? time: max_time;
  min_time = time < min_time ? time: min_time;
}

void NormalizeW(real *w_local) {
  int a;
  real w_sum = 0;
  for (a = 0; a < layer1_size; a++) w_sum += w_local[a] * w_local[a];
  w_sum = sqrt(w_sum + 1e-9);
  for (a = 0; a < layer1_size; a++) w_local[a] /= (w_sum);
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
/*
   void ReadWord(char *word, FILE *fin) {
   int a = 0, ch;
   while (!feof(fin)) {
   ch = fgetc(fin);
   if (ch == 13) continue;
   if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
   if (a > 0) {
   if (ch == '\n') ungetc(ch, fin);
   break;
   }
   if (ch == '\n') {
   strcpy(word, (char *)"</s>");
   return;
   } else continue;
   }
   word[a] = ch;
   a++;
   if (a >= MAX_STRING - 1) a--;   // Truncate too long words
   }
   word[a] = 0;
   }
   */
// Returns hash value of a word
unsigned int GetWordHash(long long word) {
	return word % vocab_hash_size;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
long long SearchVocab(struct vocab_word *vocab, int *vocab_hash, long long word) {
	unsigned int hash = GetWordHash(word);
	while (1) {
		if (vocab_hash[hash] == -1) return -1;
		if (vocab[vocab_hash[hash]].word == word) return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	return -1;
}

// Reads a word and returns its index in the vocabulary
/*
   int ReadWordIndex(FILE *fin) {
   char word[MAX_STRING];
   ReadWord(word, fin);
   if (feof(fin)) return -1;
   return SearchVocab(word);
   }
   */

struct vocab_word* ReAllocate(struct vocab_word* vocab, long long vocab_size, long long* vocab_max_size) {
	// Reallocate memory if needed
	struct vocab_word* newvocab = vocab;
	if (vocab_size + 2 >= (*vocab_max_size)) {
		(*vocab_max_size) += 1000;
		//printf("reallocate to %lld\n", (*vocab_max_size));
		newvocab = (struct vocab_word *)realloc(vocab, (*vocab_max_size) * sizeof(struct vocab_word));
		//printf("reallocated finished!");
	}
	return newvocab;
}

// Adds a word to the vocabulary
long long AddWordToVocab(struct vocab_word *vocab, int *vocab_hash, long long* vocab_size, long long* vocab_max_size, long long word) {
	unsigned int hash;
	vocab[*vocab_size].word = word;
	vocab[*vocab_size].cn = 0;
	(*vocab_size)++;
	hash = GetWordHash(word);
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = (*vocab_size) - 1;
	return (*vocab_size) - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab(struct vocab_word *vocab, int *vocab_hash, long long* vocab_size, long long* train_words) {
	int a, size;
	unsigned int hash;
	// we have no </s>
	qsort(vocab, *vocab_size, sizeof(struct vocab_word), VocabCompare);
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	size = (*vocab_size);
	(*train_words) = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if (vocab[a].cn < min_count) {
			(*vocab_size)--;
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			hash=GetWordHash(vocab[a].word);
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			vocab_hash[hash] = a;
			(*train_words) += vocab[a].cn;
		}
	}
	vocab = (struct vocab_word *)realloc(vocab, ((*vocab_size) + 1) * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	for (a = 0; a < (*vocab_size); a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab(struct vocab_word *vocab, int *vocab_hash, long long* vocab_size, long long* train_words, int* min_reduce) {
	int a, b = 0;
	unsigned int hash;
	(*train_words) = 0;
	for (a = 0; a < (*vocab_size); a++) if (vocab[a].cn > (*min_reduce)) {
		vocab[b].cn = vocab[a].cn;
		vocab[b].word = vocab[a].word;
		b++;
		(*train_words) += vocab[a].cn;
	}
	(*vocab_size) = b;
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < (*vocab_size); a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	(*min_reduce)++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree(struct vocab_word* vocab, long long vocab_size) {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (a = 0; a < vocab_size - 1; a++) {
		// First, find two smallest nodes 'min1, min2'
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			} else {
				min1i = pos2;
				pos2++;
			}
		} else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			} else {
				min2i = pos2;
				pos2++;
			}
		} else {
			min2i = pos2;
			pos2++;
		}
		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2) break;
		}
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;
		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}


long long Split(char *buffer, long long *content) {
	long long n = 0;
	char *token, *saveptr;
	for (;;buffer = NULL) {
		token = strtok_r(buffer, " ", &saveptr);
		if (token == NULL) break;
		content[n++] = atoll(token);
	}
	return n;
}
void LearnVocabFromTrainFile() {
	FILE *fin;
	long long a, i, j, n;
	long long entity, action, time;
	long long *content = (long long *)calloc(MAX_RW_LENGTH * 2 + 1, sizeof(long long));
	char *buffer = (char*)calloc(MAX_RW_LINE_LENGTH, sizeof(char));
	for (a = 0; a < vocab_hash_size; a++) {
		vocab_hash_entity[a] = -1;
		vocab_hash_action[a] = -1;
	}
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size_entity = 0;
	vocab_size_action = 0;
	while (fgets(buffer, MAX_RW_LINE_LENGTH, fin) != NULL) {
		size_t len = strlen(buffer);
		//printf("%d\n", len);
		if (len>0 && buffer[len-1]=='\n')
			buffer[len-1] = '\0';
		n = Split(buffer, content);
		//printf("%d\n", n);
		if (!n) continue;
		if (n%3 != 0) {
			printf("ERROR: training data file wrong format!\n");
			exit(1);
		}
		for (j = 0; j < n; j += 3) {
			entity = content[j];
			action = content[j + 1];
      time = content[j + 2];
			train_entity++;
			train_action++;
			if ((debug_mode > 1) && (train_entity % 100000 == 0)) {
				printf("%lldK\n", train_entity / 1000);
				fflush(stdout);
			}

      UpdateTimeBound(time);

      // add entity to vocab_entity
			i = SearchVocab(vocab_entity, vocab_hash_entity, entity);
			if (i == -1) {
				a = AddWordToVocab(vocab_entity, vocab_hash_entity, &vocab_size_entity, &vocab_max_size_entity, entity);
				vocab_entity = ReAllocate(vocab_entity, vocab_size_entity, &vocab_max_size_entity);
				vocab_entity[a].cn = 1;
			} else vocab_entity[i].cn++;

			if (vocab_size_entity > vocab_hash_size * 0.7)
				ReduceVocab(vocab_entity, vocab_hash_entity, &vocab_size_entity, &train_entity, &min_reduce_entity);
			// add action to vocab_action
			i = SearchVocab(vocab_action, vocab_hash_action, action);
			if (i == -1) {
				a = AddWordToVocab(vocab_action, vocab_hash_action, &vocab_size_action, &vocab_max_size_action, action);
				vocab_action = ReAllocate(vocab_action, vocab_size_action, &vocab_max_size_action);
				vocab_action[a].cn = 1;
			} else vocab_action[i].cn++;
			if (vocab_size_action > vocab_hash_size * 0.7)
				ReduceVocab(vocab_action, vocab_hash_action, &vocab_size_action, &train_action, &min_redue_action);
		}
	}
	SortVocab(vocab_entity, vocab_hash_entity, &vocab_size_entity, &train_entity);
	SortVocab(vocab_action, vocab_hash_action, &vocab_size_action, &train_action);
  if (timestep_size == -1) timestep_size = max_time - min_time;
	if (debug_mode > 0) {
		printf("Vocab_entity size : %lld\n", vocab_size_entity);
		printf("Vocab_action size : %lld\n", vocab_size_action);
		printf("Entity in train file: %lld\n", train_entity);
		printf("Action in train file: %lld\n", train_action);
    printf("Total timesteps from raw data: %lld\n", max_time - min_time);
	}
	train_behavior = train_entity;
	file_size = ftell(fin);
	fclose(fin);
	free(content);
	free(buffer);
	printf("LearnVocabFromTrainFile Completed\n");
}


void InitNet() {
	printf("Entering InitNet\n");
	long long a, b;
	unsigned long long next_random = 1;

  if (debug_mode > 2) {
    printf("address of syn0_action: %x\n", syn0_action);
    printf("address of vocab_action: %x\n", vocab_action);
  }
  posix_memalign((void **)&syn0_action, 128, (long long)vocab_size_action * layer1_size * sizeof(real));
  if (syn0_action == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (debug_mode > 2) {
    printf("address of syn0_action: %x\n", syn0_action);
    printf("address of vocab_action: %x\n", vocab_action);
  }
  if (hs) {
    a = posix_memalign((void **)&syn1_action, 128, (long long)vocab_size_action * layer1_size * sizeof(real));
    if (syn1_action == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size_action; a++) for (b = 0; b < layer1_size; b++)
        syn1_action[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg_action, 128, (long long)vocab_size_action * layer1_size * sizeof(real));
    if (syn1neg_action == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size_action; a++) for (b = 0; b < layer1_size; b++)
        syn1neg_action[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size_action; a++) for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      syn0_action[a * layer1_size + b] = (real)(((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }

	a = posix_memalign((void **)&syn0_entity, 128, (long long)vocab_size_entity * layer1_size * sizeof(real));
	if (syn0_entity == NULL) {printf("Memory allocation failed\n"); exit(1);}
	if (hs) {
		a = posix_memalign((void **)&syn1_entity, 128, (long long)vocab_size_entity * layer1_size * sizeof(real));
		if (syn1_entity == NULL) {printf("Memory allocation failed\n"); exit(1);}
		for (a = 0; a < vocab_size_entity; a++) for (b = 0; b < layer1_size; b++)
			syn1_entity[a * layer1_size + b] = 0;
	}
	if (negative>0) {
		a = posix_memalign((void **)&syn1neg_entity, 128, (long long)vocab_size_entity * layer1_size * sizeof(real));
		if (syn1neg_entity == NULL) {printf("Memory allocation failed\n"); exit(1);}
		for (a = 0; a < vocab_size_entity; a++) for (b = 0; b < layer1_size; b++)
			syn1neg_entity[a * layer1_size + b] = 0;
	}
	for (a = 0; a < vocab_size_entity; a++) for (b = 0; b < layer1_size; b++) {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		syn0_entity[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
	}


  // allocate memory for w
  a = posix_memalign((void **)&w_entity, 128, (long long)timestep_size * layer1_size * sizeof(real));
  if (w_entity == NULL) {printf("Memory allocation failed\n"); exit(1);}
  a = posix_memalign((void **)&w_action, 128, (long long)timestep_size * layer1_size * sizeof(real));
  if (w_action == NULL) {printf("Memory allocation failed\n"); exit(1);}

  // initialize w with random numbers
  for (a = 0; a < timestep_size; a++) {
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long) 25214903917 + 11;
      w_entity[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
      next_random = next_random * (unsigned long long) 25214903917 + 11;
      w_action[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
    }
    NormalizeW(w_entity + a * layer1_size);
    NormalizeW(w_action + a * layer1_size);
  }

  if (debug_mode > 1) {
    printf("Completed InitNet\n");
  }
	CreateBinaryTree(vocab_entity, vocab_size_entity);
	CreateBinaryTree(vocab_action, vocab_size_action);
  if (debug_mode > 1) {
    printf("Completed CreateBinaryTree\n");
  }
}

void *TrainModelThread(void *id) {
	long long a, b, d, cw, entity, action, time, word, last_word, sentence_length = 0, sentence_position = 0;
	long long behavior_count = 0, last_behavior_count = 0;
	long long l1, l2, c, target, label, local_iter = iter;
	unsigned long long next_random = (long long)id;
  real m_ti, m_tj, m_si, m_sj, m_ts;              // variables to store temp values
  long long t, s;

	real f, g;
	clock_t now;
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  real *wt_e = (real *)calloc(layer1_size, sizeof(real));     // error propagation variable for wt
  real *ws_e = (real *)calloc(layer1_size, sizeof(real));     // error propagation variable for ws
  real *wt_action = (real *)calloc(layer1_size, sizeof(real));
  real *ws_action = (real *)calloc(layer1_size, sizeof(real));
  real *wt_entity = (real *)calloc(layer1_size, sizeof(real));
  real *ws_entity = (real *)calloc(layer1_size, sizeof(real));
	FILE *fi = fopen(train_file, "rb");
	char* buffer = (char *)calloc(MAX_RW_LINE_LENGTH, sizeof(char));
	long long *content = (long long *)calloc(MAX_RW_LENGTH * 2 + 1, sizeof(long long));
	//long long *sen_entity = (long long *)calloc(MAX_RW_LENGTH, sizeof(long long));
	//long long *sen_action = (long long *)calloc(MAX_RW_LENGTH, sizeof(long long));
	long long *sen = (long long *)calloc(MAX_RW_LENGTH * 2, sizeof(long long));
	long long n, i;
	long long input_is_action, output_is_action;
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);    // how to guarantee fi points to entity ?
	fgets(buffer, MAX_RW_LINE_LENGTH, fi);
	if (debug_mode > 0) printf("thread %d\n", id);
	while (1) {
    // display progress
		if (behavior_count - last_behavior_count> 10000) {
			behavior_count_actual += behavior_count - last_behavior_count;
			last_behavior_count = behavior_count;
			if ((debug_mode > 1)) {
				now=clock();
				printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
						behavior_count_actual / (real)(iter * train_entity + 1) * 100,
						behavior_count_actual / ((real)(now - start + 2) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			alpha = starting_alpha * (1 - behavior_count_actual / (real)(iter * train_entity + 1));
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		}
    // build sentence
		if (sentence_length == 0 && fgets(buffer, MAX_RW_LINE_LENGTH, fi) != NULL) {
			size_t len = strlen(buffer);
			if (len>0 && buffer[len-1]=='\n') buffer[len-1] = '\0';
			n = Split(buffer, content);
			if (n % 3 != 0) {
				printf("wrong format!");
        exit(1);
			}
			sen[0] = -1;
      // content[] starts at entity
			for (i=0; i<n; i += 3) {
				entity = SearchVocab(vocab_entity, vocab_hash_entity, content[i]);
				if (entity == -1) continue;
				action = SearchVocab(vocab_action, vocab_hash_action, content[i + 1]);
				if (action == -1) continue;
        time = content[i + 2];
				sen[sentence_length++] = entity;
				sen[sentence_length++] = action;
        sen[sentence_length++] = time;
        // sentence_length is set to be multiple of 4, so that judging an item is entity or action or time can be done
        // efficiently using bit operation
        sentence_length++;
			}
			behavior_count += sentence_length / 4;
			sentence_position = 0;
			if (debug_mode > 2) {
				printf("new sentence with length=%lld behavior_count=%lld n=%lld\n", sentence_length, behavior_count, n);
			}
		}
		if (feof(fi) || (behavior_count > train_behavior / num_threads + 100)) {
			behavior_count_actual += behavior_count - last_behavior_count;
			local_iter--;
			if (debug_mode > 1)
				printf("thread %d local_iter %d\n", id, local_iter);
			if (local_iter == 0) break;
			behavior_count = 0;
			last_behavior_count = 0;
			sentence_length = 0;
			fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
			fgets(buffer, MAX_RW_LINE_LENGTH, fi);
			continue;
		}
		word = sen[sentence_position];
    if (sentence_position % 4 == 0)
      s = sen[sentence_position + 2];
    else if (sentence_position % 4 == 1)
      s = sen[sentence_position + 1];
    else
      printf("Alignment error");
    // Copy ws from matrix w
    for (c = 0; c < layer1_size; c++) {
      ws_entity[c] = w_entity[c + s];
      ws_action[c] = w_action[c + s];
    }
    NormalizeW(ws_entity);
    NormalizeW(ws_action);

		if (debug_mode > 2) {
			printf("sentence_position(output)=%d word=%lld s=%lld\n", sentence_position, word, s);
		}
		if (word == -1) continue;
		for (c = 0; c < layer1_size; c++) neu1[c] = 0;
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		next_random = next_random * (unsigned long long)25214903917 + 11;
		b = next_random % window;
		if (cbow) {  //train the cbow architecture
			// in -> hidden
			cw = 0;
			output_is_action = sentence_position & 1;
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				c = sentence_position - window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				input_is_action = c & 1;
				if (!input_is_action) {
					for (c = 0; c < layer1_size; c++) neu1[c] += syn0_entity[c + last_word * layer1_size];
				} else {
					for (c = 0; c < layer1_size; c++) neu1[c] += syn0_action[c + last_word * layer1_size];
				}
				cw++;
			}
			if (cw) {
				for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
				if (hs) {
					// For entity
					if (!output_is_action) {
						for (d = 0; d < vocab_entity[word].codelen; d++) {
							f = 0;
							l2 = vocab_entity[word].point[d] * layer1_size;
							// Propagate hidden -> output
							for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1_entity[c + l2];
							if (f <= -MAX_EXP) continue;
							else if (f >= MAX_EXP) continue;
							else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
							// 'g' is the gradient multiplied by the learning rate
							g = (1 - vocab_entity[word].code[d] - f) * alpha;
							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1_entity[c + l2];
							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++) syn1_entity[c + l2] += g * neu1[c];
						}
					} else {
						// For action
						for (d = 0; d < vocab_action[word].codelen; d++) {
							f = 0;
							l2 = vocab_action[word].point[d] * layer1_size;
							// Propagate hidden -> output
							for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1_action[c + l2];
							if (f <= -MAX_EXP) continue;
							else if (f >= MAX_EXP) continue;
							else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
							// 'g' is the gradient multiplied by the learning rate
							g = (1 - vocab_action[word].code[d] - f) * alpha;
							// Propagate errors output -> hidden
							for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1_action[c + l2];
							// Learn weights hidden -> output
							for (c = 0; c < layer1_size; c++) syn1_action[c + l2] += g * neu1[c];
						}
					}
				}
				// NEGATIVE SAMPLING
				if (negative > 0) {
					//For entity
					if (!output_is_action) {
						for (d = 0; d < negative + 1; d++) {
							if (d == 0) {
								target = word;
								label = 1;
							} else {
								next_random = next_random * (unsigned long long)25214903917 + 11;
								target = table_entity[(next_random >> 16) % table_size];
								//if (target == 0) target = next_random % (vocab_size - 1) + 1;
								if (target == word) continue;
								label = 0;
							}
							l2 = target * layer1_size;
							f = 0;
							for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg_entity[c + l2];
							if (f > MAX_EXP) g = (label - 1) * alpha;
							else if (f < -MAX_EXP) g = (label - 0) * alpha;
							else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
							for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg_entity[c + l2];
							for (c = 0; c < layer1_size; c++) syn1neg_entity[c + l2] += g * neu1[c];
						}
					} else {
						// For action
						for (d = 0; d < negative + 1; d++) {
							if (d == 0) {
								target = word;
								label = 1;
							} else {
								next_random = next_random * (unsigned long long)25214903917 + 11;
								target = table_action[(next_random >> 16) % table_size];
								//if (target == 0) target = next_random % (vocab_size - 1) + 1;
								if (target == word) continue;
								label = 0;
							}
							l2 = target * layer1_size;
							f = 0;
							for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg_action[c + l2];
							if (f > MAX_EXP) g = (label - 1) * alpha;
							else if (f < -MAX_EXP) g = (label - 0) * alpha;
							else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
							for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg_action[c + l2];
							for (c = 0; c < layer1_size; c++) syn1neg_action[c + l2] += g * neu1[c];
						}
					}
				}
				// hidden -> in
				for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
					c = sentence_position - window + a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = sen[c];
					input_is_action = c & 1;
					if (!input_is_action) {
						for (c = 0; c < layer1_size; c++) syn0_entity[c + last_word * layer1_size] += neu1e[c];
					} else {
						for (c = 0; c < layer1_size; c++) syn0_action[c + last_word * layer1_size] += neu1e[c];
					}
				}
			}
		} else {  //train skip-gram
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          //input c output at sentence_position
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          if (c % 4 == 3 || c % 4 == 2) continue;
          last_word = sen[c];
          l1 = last_word * layer1_size;
          input_is_action = (c % 4) == 1;
          output_is_action = (sentence_position % 4) == 1;
          if (input_is_action)
            t = sen[c + 1];
          else
            t = sen[c + 2];

          if (debug_mode > 2) {
            printf("input position: %lld, output position: %lld, t: %lld, s: %lld\n", c, sentence_position, t, s);
            printf("input word: %lld, output word: %lld, input_action: %d, output action: %d\n", last_word, word, input_is_action, output_is_action);
          }

          // Copy ws from matrix w
          for (c = 0; c < layer1_size; c++) {
            wt_entity[c] = w_entity[c + t];
            wt_action[c] = w_action[c + t];
          }
          NormalizeW(wt_entity);
          NormalizeW(wt_action);



          for (c = 0; c < layer1_size; c++) neu1e[c] = wt_e[c] = ws_e[c] = 0;
          // HIERARCHICAL SOFTMAX
          if (hs) {
            // For out: entity
            if (!output_is_action) {
              for (d = 0; d < vocab_entity[word].codelen; d++) {
                f = 0;
                m_si = m_sj = m_ti = m_tj = 0;
                l2 = vocab_entity[word].point[d] * layer1_size;
                // Propagate hidden -> output
                if (!input_is_action) {   // in: entity, out: entity
                  for (c = 0; c < layer1_size; c++) {
                    f += syn0_entity[c + l1] * syn1_entity[c + l2];
                    m_ti += wt_entity[c] * syn0_entity[c + l1];
                    m_tj += wt_entity[c] * syn1_entity[c + l2];
                    m_si += ws_entity[c] * syn0_entity[c + l1];
                    m_sj += ws_entity[c] * syn1_entity[c + l2];
                    m_ts += ws_entity[c] * wt_entity[c];
                  }
                } else {    // in: action, out: entity
                  for (c = 0; c < layer1_size; c++) {
                    f += syn0_action[c + l1] * syn1_entity[c + l2];
                    m_ti += wt_entity[c] * syn0_action[c + l1];
                    m_tj += wt_entity[c] * syn1_entity[c + l2];
                    m_si += ws_entity[c] * syn0_action[c + l1];
                    m_sj += ws_entity[c] * syn1_entity[c + l2];
                    m_ts += ws_entity[c] * wt_entity[c];
                  }
                }
                f += - m_ti * m_tj - m_si * m_sj + m_ts * m_ti * m_sj;
                if (f <= -MAX_EXP) continue;
                else if (f >= MAX_EXP) continue;
                else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                // 'g' is the gradient multiplied by the learning rate
                g = (1 - vocab_entity[word].code[d] - f) * alpha;
                // Propagate errors output -> hidden
                if (!input_is_action) {   // in: entity, out: entity
                  for (c = 0; c < layer1_size; c++) {
                    neu1e[c] += g * (syn1_entity[c + l2] - m_tj * wt_entity[c] - m_sj * ws_entity[c] + m_ts * m_sj * wt_entity[c]);
                  }
                } else {    // in: action, out: entity
                  for (c = 0; c < layer1_size; c++) {
                    neu1e[c] += g * (syn1_entity[c + l2] - m_tj * wt_action[c] - m_sj * ws_entity[c] + m_ts * m_sj * wt_action[c]);
                  }
                }
                // Propagate errors wt, ws
                if (s == t) {
                  if (!input_is_action) { // in: entity, out: entity
                    for (c = 0; c < layer1_size; c++) {
                      wt_e[c] += -g * (m_ti * syn1_entity[c + l2] + m_tj * syn0_entity[c + l1]);
                    }
                  } else {  // in: action, out: entity
                    for (c = 0; c < layer1_size; c++) {
                      wt_e[c] += -g * (m_ti * syn1_entity[c + l2] + m_tj * syn0_action[c + l1]);
                    }
                  }
                } else {
                  if (!input_is_action) { // in: entity, out: entity
                    for (c = 0; c < layer1_size; c++) {
                      wt_e[c] += g * (m_sj * (m_ts * syn0_entity[c + l1] + m_ti * ws_entity[c]) - m_tj * syn0_entity[c + l1] - m_ti * syn1_entity[c + l1]);
                      ws_e[c] += g * (m_ti * (m_ts * syn1_entity[c + l2] + m_sj * wt_entity[c]) - m_si * syn1_entity[c + l2] - m_sj * syn0_entity[c + l1]);
                    }
                  } else {  // in: action, out: entity
                    for (c = 0; c < layer1_size; c++) {
                      wt_e[c] += g * (m_sj * (m_ts * syn0_action[c + l1] + m_ti * ws_entity[c]) - m_tj * syn0_action[c + l1] - m_ti * syn1_entity[c + l1]);
                      ws_e[c] += g * (m_ti * (m_ts * syn1_entity[c + l2] + m_sj * wt_action[c]) - m_si * syn1_entity[c + l2] - m_sj * syn0_action[c + l1]);
                    }
                  }
                }

                // Learn weights hidden -> output
                if (!input_is_action) {   // in: entity, out: entity
                  for (c = 0; c < layer1_size; c++)
                    syn1_entity[c + l2] += g * (syn0_entity[c + l1] - m_ti * wt_entity[c] - m_si * ws_entity[c] + m_ti * m_ts * ws_entity[c]);
                } else {
                  for (c = 0; c < layer1_size; c++) // in: action, out: entity
                    syn1_entity[c + l2] += g * (syn0_action[c + l1] - m_ti * wt_action[c] - m_si * ws_entity[c] + m_ti * m_ts * ws_entity[c]);
                }
              } // end enumerate Huffman tree
            } else {
              // For out: action
              for (d = 0; d < vocab_action[word].codelen; d++) {
                f = 0;
                m_si = m_sj = m_ti = m_tj = 0;
                l2 = vocab_action[word].point[d] * layer1_size;
                // Propagate hidden -> output
                if (!input_is_action) {   // in(wt, syn0): entity, out(ws, syn1): action
                  for (c = 0; c < layer1_size; c++) {
                    f += syn0_entity[c + l1] * syn1_action[c + l2];
                    m_ti += wt_entity[c] * syn0_entity[c + l1];
                    m_tj += wt_entity[c] * syn1_action[c + l2];
                    m_si += ws_action[c] * syn0_entity[c + l1];
                    m_sj += ws_action[c] * syn1_action[c + l2];
                    m_ts += ws_action[c] * wt_entity[c];
                  }
                } else {
                  for (c = 0; c < layer1_size; c++) {   // in(wt, syn0): action, out(ws, syn1): action
                    f += syn0_action[c + l1] * syn1_action[c + l2];
                    m_ti += wt_action[c] * syn0_action[c + l1];
                    m_tj += wt_action[c] * syn1_action[c + l2];
                    m_si += ws_action[c] * syn0_action[c + l1];
                    m_sj += ws_action[c] * syn1_action[c + l2];
                    m_ts += ws_action[c] * wt_action[c];
                  }
                }
                f += - m_ti * m_tj - m_si * m_sj + m_ts * m_ti * m_sj;
                if (f <= -MAX_EXP) continue;
                else if (f >= MAX_EXP) continue;
                else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                // 'g' is the gradient multiplied by the learning rate
                g = (1 - vocab_action[word].code[d] - f) * alpha;
                // Propagate errors output -> hidden
                if (!input_is_action) {   // in(wt, syn0): entity, out(ws, syn1): action
                  for (c = 0; c < layer1_size; c++) {
                    neu1e[c] += g * (syn1_action[c + l2] - m_tj * wt_entity[c] - m_sj * ws_action[c] + m_ts * m_sj * wt_entity[c]);
                  }
                } else {  // in(wt, syn0): action, out(ws, syn1): action
                  for (c = 0; c < layer1_size; c++) {
                    neu1e[c] += g * (syn1_action[c + l2] - m_tj * wt_action[c] - m_sj * ws_action[c] + m_ts * m_sj * wt_action[c]);
                  }
                }
                // Propagate errors wt, ws
                if (s == t) {
                  if (!input_is_action) { // in(wt, syn0): entity, out(ws, syn1): action
                    for (c = 0; c < layer1_size; c++) {
                      wt_e[c] += -g * (m_ti * syn1_action[c + l2] + m_tj * syn0_entity[c + l1]);
                    }
                  } else {  // in(wt, syn0): action, out(ws, syn1): action
                    for (c = 0; c < layer1_size; c++) {
                      wt_e[c] += -g * (m_ti * syn1_action[c + l2] + m_tj * syn0_action[c + l1]);
                    }
                  }
                } else {
                  if (!input_is_action) { // in(wt, syn0): entity, out(ws, syn1): action
                    for (c = 0; c < layer1_size; c++) {
                      wt_e[c] += g * (m_sj * (m_ts * syn0_entity[c + l1] + m_ti * ws_action[c]) - m_tj * syn0_entity[c + l1] - m_ti * syn1_action[c + l1]);
                      ws_e[c] += g * (m_ti * (m_ts * syn1_action[c + l2] + m_sj * wt_entity[c]) - m_si * syn1_action[c + l2] - m_sj * syn0_entity[c + l1]);
                    }
                  } else {  // in(wt, syn0): action, out(ws, syn1): action
                    for (c = 0; c < layer1_size; c++) {
                      wt_e[c] += g * (m_sj * (m_ts * syn0_action[c + l1] + m_ti * ws_action[c]) - m_tj * syn0_action[c + l1] - m_ti * syn1_action[c + l1]);
                      ws_e[c] += g * (m_ti * (m_ts * syn1_action[c + l2] + m_sj * wt_action[c]) - m_si * syn1_action[c + l2] - m_sj * syn0_action[c + l1]);
                    }
                  }
                }
                // Learn weights hidden -> output
                if (!input_is_action) {   // in(wt, syn0): entity, out(ws, syn1): action
                  for (c = 0; c < layer1_size; c++)
                    syn1_action[c + l2] += g * (syn0_entity[c + l1] - m_ti * wt_entity[c] - m_si * ws_action[c] + m_ti * m_ts * ws_action[c]);
                } else {  // in(wt, syn0): action, out(ws, syn1): action
                  for (c = 0; c < layer1_size; c++)
                    syn1_action[c + l2] += g * (syn0_action[c + l1] - m_ti * wt_action[c] - m_si * ws_action[c] + m_ti * m_ts * ws_action[c]);
                }
              } // end enumerate Huffman tree
					}
				}   // end HIERARCHICAL SOFTMAX
				// NEGATIVE SAMPLING
				if (negative > 0) {
					// For entity
					if (!output_is_action) {
						for (d = 0; d < negative + 1; d++) {
							if (d == 0) {
								target = word;
								label = 1;
							} else {
								next_random = next_random * (unsigned long long)25214903917 + 11;
								target = table_entity[(next_random >> 16) % table_size];
								//if (target == 0) target = next_random % (vocab_size - 1) + 1;
								if (target == word) continue;
								label = 0;
							}
							l2 = target * layer1_size;
							f = 0;
							if (!input_is_action) {
								for (c = 0; c < layer1_size; c++) f += syn0_entity[c + l1] * syn1neg_entity[c + l2];
							} else {
								for (c = 0; c < layer1_size; c++) f += syn0_action[c + l1] * syn1neg_entity[c + l2];
							}
							if (f > MAX_EXP) g = (label - 1) * alpha;
							else if (f < -MAX_EXP) g = (label - 0) * alpha;
							else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
							for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg_entity[c + l2];
							if (!input_is_action) {
								for (c = 0; c < layer1_size; c++) syn1neg_entity[c + l2] += g * syn0_entity[c + l1];
							} else {
								for (c = 0; c < layer1_size; c++) syn1neg_entity[c + l2] += g * syn0_action[c + l1];
							}
						}
					} else {
						for (d = 0; d < negative + 1; d++) {
							if (d == 0) {
								target = word;
								label = 1;
							} else {
								next_random = next_random * (unsigned long long)25214903917 + 11;
								target = table_action[(next_random >> 16) % table_size];
								//if (target == 0) target = next_random % (vocab_size - 1) + 1;
								if (target == word) continue;
								label = 0;
							}
							l2 = target * layer1_size;
							f = 0;
							if (!input_is_action) {
								for (c = 0; c < layer1_size; c++) f += syn0_entity[c + l1] * syn1neg_action[c + l2];
							} else {
								for (c = 0; c < layer1_size; c++) f += syn0_action[c + l1] * syn1neg_action[c + l2];
							}
							if (f > MAX_EXP) g = (label - 1) * alpha;
							else if (f < -MAX_EXP) g = (label - 0) * alpha;
							else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
							for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg_action[c + l2];
							if (!input_is_action) {
								for (c = 0; c < layer1_size; c++) syn1neg_action[c + l2] += g * syn0_entity[c + l1];
							} else {
								for (c = 0; c < layer1_size; c++) syn1neg_action[c + l2] += g * syn0_action[c + l1];
							}
						}
					}
				}
				// Learn weights input -> hidden, wt
				if (!input_is_action) {   // in(wt, syn0): entity, out(ws, syn1): ?
					for (c = 0; c < layer1_size; c++) {
            syn0_entity[c + l1] += neu1e[c];
            w_entity[c + t] += wt_e[c];
          }
				} else {  // in(wt, syn0): action, out(ws, syn1): ?
					for (c = 0; c < layer1_size; c++) {
            syn0_action[c + l1] += neu1e[c];
            w_action[c + t] += wt_e[c];
          }
				}
        // Learn ws
        if (s != t) { // perheps this if is not necessary, as ws_e will be 0 if s != t
          if (!output_is_action) {  // in: ?, out: entity
            for (c = 0; c < layer1_size; c++) {
              w_entity[c + s] += ws_e[c];
            }
          } else {
            for (c = 0; c < layer1_size; c++) {
              w_action[c + s] += ws_e[c];
            }
          }
        } // end if s != t
			} // end enumerate input
		} // end skip-gram
		sentence_position++;
    if (sentence_position % 4 == 3) sentence_position++;
    else if (sentence_position % 4 == 2) sentence_position++;
		if (sentence_position >= sentence_length) {
			sentence_length = 0;
			continue;
		}
	}
	fclose(fi);
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}

void TrainModel() {
	long a, b, c, d;
	FILE *fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;
	//if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
	//if (save_vocab_file[0] != 0) SaveVocab();
	LearnVocabFromTrainFile();
	if (output_file_entity[0] == 0) return;
	if (output_file_action[0] == 0) return;
  if (output_file_entity_time[0] == 0) return;
  if (output_file_action_time[0] == 0) return;
	InitNet();
	if (negative > 0) {
		table_entity = InitUnigramTable(vocab_entity, vocab_size_entity);
		table_action = InitUnigramTable(vocab_action, vocab_size_action);
		printf("InitUnigramTable\n");
	}
	start = clock();
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	printf("Starting saving model\n");
	fo = fopen(output_file_entity, "wb");
	// Save the word vectors
	fprintf(fo, "%lld %lld\n", vocab_size_entity, layer1_size);
	for (a = 0; a < vocab_size_entity; a++) {
		fprintf(fo, "%lld ", vocab_entity[a].word);
		for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0_entity[a * layer1_size + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
	fo = fopen(output_file_action, "wb");
	fprintf(fo, "%lld %lld\n", vocab_size_action, layer1_size);
	for (a = 0; a < vocab_size_action; a++) {
		fprintf(fo, "%lld ", vocab_action[a].word);
		for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0_action[a * layer1_size + b]);
		fprintf(fo, "\n");
	}
	fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output_entity <file>\n");
		printf("\t\tUse <file> to save the resulting entity vectors\n");
		printf("\t-output_action <file>\n");
		printf("\t\tUse <file> to save the resulting action vectors\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of entity/action vectors; default is 100\n");
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		//    printf("\t-sample <float>\n");
		//    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
		//    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 12)\n");
		printf("\t-iter <int>\n");
		printf("\t\tRun more training iterations (default 5)\n");
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
		//    printf("\t-classes <int>\n");
		//    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		//    printf("\t-binary <int>\n");
		//    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		//    printf("\t-save-vocab <file>\n");
		//    printf("\t\tThe vocabulary will be saved to <file>\n");
		//    printf("\t-read-vocab <file>\n");
		//    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-cbow <int>\n");
		printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output_entity entity.txt -output_action action.txt -size 200 -window 5 -negative 5 -hs 0 -cbow 1 -iter 3\n\n");
		return 0;
	}
	output_file_entity[0] = 0;
	output_file_action[0] = 0;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	//  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	//  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	//  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if (cbow) alpha = 0.05;
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output_entity", argc, argv)) > 0) strcpy(output_file_entity, argv[i + 1]);
	if ((i = ArgPos((char *)"-output_action", argc, argv)) > 0) strcpy(output_file_action, argv[i + 1]);
  if ((i = ArgPos((char *)"-output_entity_time", argc, argv)) > 0) strcpy(output_file_entity_time, argv[i + 1]);
  if ((i = ArgPos((char *)"-output_action_time", argc, argv)) > 0) strcpy(output_file_action_time, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = 2 * atoi(argv[i + 1]);
	//  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-timestep", argc, argv)) > 0) timestep_size = atoi(argv[i + 1]);
	//  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
	vocab_entity = (struct vocab_word *)calloc(vocab_max_size_entity, sizeof(struct vocab_word));
	vocab_hash_entity = (int *)calloc(vocab_hash_size, sizeof(int));
	vocab_action = (struct vocab_word *)calloc(vocab_max_size_action, sizeof(struct vocab_word));
  if (debug_mode > 1) printf("address of vocab_action: %x\n", vocab_action);
	vocab_hash_action= (int *)calloc(vocab_hash_size, sizeof(int));
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
	}
	printf("Argument Initialized\n");
	TrainModel();
	return 0;
}
