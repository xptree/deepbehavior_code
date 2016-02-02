#include "Graph.h"
#include "Util.h"
#include "Parameter.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "log4cpp/Category.hh"
#include "log4cpp/OstreamAppender.hh"
#include "log4cpp/FileAppender.hh"
#include "log4cpp/Priority.hh"
#include "log4cpp/PatternLayout.hh"
using namespace std;


#define BUFFER_SIZE 1024

Graph* buildGraph(Parameter* para)
{
	log4cpp::Category& root = log4cpp::Category::getRoot();
	log4cpp::Category& logger = root.getInstance("logger");
	Graph* g = new Graph();
	char buffer[BUFFER_SIZE];
	int last = 1;
	long long loaded_bytes = 0;
	if (para->behavior_file.length() > 0) {
		long long behavior_file_size = Util::getFileSize(para->behavior_file.c_str());
		logger.info("start loading from %s, size of file %.2fMB", para->behavior_file.c_str(), behavior_file_size/1024./1024.);
		freopen(para->behavior_file.c_str(), "rb", stdin);
		while (fgets(buffer, BUFFER_SIZE, stdin) != NULL) {
			size_t len = strlen(buffer);
			loaded_bytes += len * sizeof(char);
			if (loaded_bytes / double(behavior_file_size) * 100 >= last) {
				logger.info("%d%% loaded ...", int(loaded_bytes / double(behavior_file_size) * 100));
				last += 10;
			}
			if (len>0 && buffer[len-1]=='\n')
				buffer[len-1] = '\0';
			vector<string> elems = Util::split(buffer, '\t');
			int user             = atoi(elems[0].c_str());
			int timestamp        = atoi(elems[1].c_str()) - 2005;
			int action           = atoi(elems[2].c_str());
			g->addBehavior(user, timestamp, action);
		}
		fclose(stdin);
	}
	last = 1;
	loaded_bytes = 0;
	if (para->entity_file.length() > 0) {
		long long entity_file_size = Util::getFileSize(para->entity_file.c_str());
		logger.info("start loading from %s, size of file %.2fMB", para->entity_file.c_str(), entity_file_size/1024./1024.);
		freopen(para->entity_file.c_str(), "rb", stdin);
		while (fgets(buffer, BUFFER_SIZE, stdin) != NULL) {
			size_t len = strlen(buffer);
			loaded_bytes += len * sizeof(char);
			if (loaded_bytes / double(entity_file_size) * 100 >= last) {
				logger.info("%d%% loaded ...", int(loaded_bytes / double(entity_file_size) * 100));
				last += 10;
			}
			if (len>0 && buffer[len-1]=='\n')
				buffer[len-1] = '\0';
			vector<string> elems = Util::split(buffer, '\t');
			int u                = atoi(elems[0].c_str());
			int v                = atoi(elems[1].c_str());
			double w                = atof(elems[2].c_str());
			g->addEntityRelationship(u, v, w);
		}
		fclose(stdin);
	}
	last = 1;
	loaded_bytes = 0;
	if (para->action_file.length() > 0) {
		logger.info(para->action_file);
		long long action_file_size = Util::getFileSize(para->action_file.c_str());
		logger.info("start loading from %s, size of file %.2fMB", para->action_file.c_str(), action_file_size/1024./1024.);
		freopen(para->action_file.c_str(), "rb", stdin);
		while (fgets(buffer, BUFFER_SIZE, stdin) != NULL) {
			size_t len = strlen(buffer);
			loaded_bytes += len * sizeof(char);
			if (loaded_bytes / double(action_file_size+1) * 100 >= last) {
				//infoCategory.info("%d%% loaded ...", int(loadedBytes / double(userFileSize) * 100));
				last += 10;
			}
			if (len>0 && buffer[len-1]=='\n')
				buffer[len-1] = '\0';
			vector<string> elems = Util::split(buffer, '\t');
			int u                = atoi(elems[0].c_str());
			int v                = atoi(elems[1].c_str());
			double w                = atof(elems[2].c_str());
			g->addActionRelationship(u, v, w);
		}
		fclose(stdin);
	}
	g->preprocess(para);
	return g;
}

int main(int argc, char** argv)
{
	//log4cpp::OstreamAppender* osAppender = new log4cpp::OstreamAppender("osAppender", &cout);
	log4cpp::Appender *osAppender = new log4cpp::FileAppender("default", "program.log", false);
	log4cpp::PatternLayout* pLayout = new log4cpp::PatternLayout();
	pLayout->setConversionPattern("%d: %p %c %x: %m%n");
	osAppender->setLayout(pLayout);
	log4cpp::Category& root = log4cpp::Category::getRoot();
	log4cpp::Category& logger = root.getInstance("logger");
	logger.addAppender(osAppender);
	logger.setPriority(log4cpp::Priority::INFO);
	Parameter* para = new Parameter();
	para->parse(argc, argv);
	Graph* g = buildGraph(para);
	vector<vector<int> > corpus;
	g->rw_parallel(corpus, para);
	freopen(para->output_file.c_str(), "wb", stdout);
	for (size_t i=0; i<corpus.size(); ++i) {
		if (corpus[i].size() <= 1) continue;
		for (size_t j=0; j<corpus[i].size(); ++j)
			printf("%d%c", corpus[i][j], j+1<corpus[i].size() ? ' ':'\n');
	}
	fclose(stdout);
	return 0;
}
