#include "Parameter.h"
#include <cstdlib>
#include <cstring>
#include "log4cpp/Category.hh"
#include "log4cpp/OstreamAppender.hh"
#include "log4cpp/FileAppender.hh"
#include "log4cpp/Priority.hh"
#include "log4cpp/PatternLayout.hh"

Parameter::Parameter()
{
	setDefault();
}

Parameter::~Parameter() {}

int Parameter::argPos(char* str, int argc, char** argv) {
	for (int i=0; i<argc; ++i) {
		if (!strcmp(str, argv[i])) {
			if (i == argc-1) {
				log4cpp::Category& root = log4cpp::Category::getRoot();
				log4cpp::Category& logger = root.getInstance("logger");
				logger.info("Argument missing for %s", str);
				exit(1);
			}
			return i;
		}
	}
	return -1;
}

void Parameter::parse(int argc, char** argv) {
	int i;
	if ((i = argPos((char *)"-entity_time_const", argc, argv)) > 0) entity_time_const = atof(argv[i + 1]);
	if ((i = argPos((char *)"-action_time_const", argc, argv)) > 0) action_time_const = atof(argv[i + 1]);

	if ((i = argPos((char *)"-entity_relation_const", argc, argv)) > 0) entity_relation_const = atof(argv[i + 1]);
	if ((i = argPos((char *)"-action_relation_const", argc, argv)) > 0) action_relation_const = atof(argv[i + 1]);

	if ((i = argPos((char *)"-neighbor_entity", argc, argv)) > 0) neighbor_entity = atof(argv[i + 1]);
	if ((i = argPos((char *)"-self_entity", argc, argv)) > 0) self_entity = atof(argv[i + 1]);
	if ((i = argPos((char *)"-neighbor_action", argc, argv)) > 0) neighbor_action = atof(argv[i + 1]);
	if ((i = argPos((char *)"-self_action", argc, argv)) > 0) self_action = atof(argv[i + 1]);

	if ((i = argPos((char *)"-num_walks_entity", argc, argv)) > 0) num_walks_entity = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-num_walks_action", argc, argv)) > 0) num_walks_action = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-walk_length", argc, argv)) > 0) walk_length = atoi(argv[i + 1]);
	if ((i = argPos((char *)"-num_threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

	if ((i = argPos((char*)"-entity_file", argc, argv)) > 0) entity_file = string(argv[i + 1]);
	if ((i = argPos((char*)"-action_file", argc, argv)) > 0) action_file = string(argv[i + 1]);
	if ((i = argPos((char*)"-behavior_file", argc, argv)) > 0) behavior_file = string(argv[i + 1]);

	if ((i = argPos((char*)"-output_file", argc, argv)) > 0) output_file = string(argv[i + 1]);
}


void Parameter::setDefault() {
	entity_time_const = 1.0;
	action_time_const = 1.0;
	entity_relation_const = 1.0;
	action_relation_const = 1.0;

	neighbor_entity = 0.25;
	self_entity = 0.25;
	neighbor_action = 0.25;
	self_action = 0.25;

	num_walks_entity = num_walks_action = 10;
	walk_length = 40;
	num_threads = 32;

	entity_file = action_file = behavior_file = "";
	output_file = "";
}

