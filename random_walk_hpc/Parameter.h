#pragma once
#include <string>
using namespace std;

class Parameter
{
public:
	Parameter();
	~Parameter();
	void parse(int argc, char** argv);
	void setDefault();
	int argPos(char* str, int argc, char** argv);
	double entity_time_const;
	double action_time_const;
	double entity_relation_const;
	double action_relation_const;

	double neighbor_entity;
	double self_entity;
	double neighbor_action;
	double self_action;

	int num_walks_entity;
	int num_walks_action;
	int walk_length;
	int num_threads;

	string entity_file;
	string action_file;
	string behavior_file;
	string output_file;
};
