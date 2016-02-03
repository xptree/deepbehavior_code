#include "Entity.h"
#include "Action.h"
#include "Behavior.h"
#include "Parameter.h"
#include "Graph.h"
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "log4cpp/Category.hh"
#include "log4cpp/OstreamAppender.hh"
#include "log4cpp/FileAppender.hh"
#include "log4cpp/Priority.hh"
#include "log4cpp/PatternLayout.hh"

Graph::Graph() {}

Graph::~Graph()
{
	for (size_t i=0; i<entity.size(); ++i)
	if (entity[i])
		delete entity[i];
	for (size_t i=0; i<action.size(); ++i)
	if (action[i])
		delete action[i];
	for (size_t i=0; i<behavior.size(); ++i)
		delete behavior[i];
}

void Graph::addEntity(int index)
{
	if (index + 1 > entity.size()) {
		entity.resize(index+1, NULL);
	}
	if (entity[index]) return;
	Entity* e = new Entity(index);
	entity[index] = e;
}

void Graph::addAction(int index)
{
	if (index + 1 > action.size()) {
		action.resize(index+1, NULL);
	}
	if (action[index]) return;
	Action* a = new Action(index);
	action[index] = a;
}

void Graph::addEntityRelationship(int u, int v, double w)
{
	addEntity(u);
	addEntity(v);
	Entity* e_u = entity[u];
	Entity* e_v = entity[v];
	e_u->addEntityRelationship(e_v, w);
	e_v->addEntityRelationship(e_u, w);
}

void Graph::addActionRelationship(int u, int v, double w)
{
	addAction(u);
	addAction(v);
	Action* a_u = action[u];
	Action* a_v = action[v];
	a_u->addActionRelationship(a_v, w);
	a_v->addActionRelationship(a_u, w);
}

void Graph::addBehavior(int e, int t, int a)
{
	addEntity(e);
	addAction(a);
	Entity* this_e = entity[e];
	Action* this_a = action[a];
	Behavior* b = new Behavior(this_e, this_a, t);
	behavior.push_back(b);
	this_e->addBehavior(b);
	this_a->addBehavior(b);
}

void Graph::rw_serial(vector<vector<int> >& corpus, Parameter* para) const
{
	log4cpp::Category& root = log4cpp::Category::getRoot();
	log4cpp::Category& logger = root.getInstance("logger");
	/*
	for (size_t i=0; i<behavior.size(); ++i) {
		for (int w=0; w<para->num_walks; ++w) {
			vector<int> sentence;
			Behavior* b = behavior[i];
			sentence.push_back(b->entity()->getId());
			sentence.push_back(b->action()->getId());
			for (int l=1; l<para->walk_length; ++l) {
				//logger.info("Now at %d %d", b->entity()->getId(), b->action()->getId());
				b = b->walk(para);
				if (b==NULL) break;
				sentence.push_back(b->entity()->getId());
				sentence.push_back(b->action()->getId());
			}
			corpus.push_back(sentence);
		}
		if (i % 100==0) printf("%d\n", i);
	}*/
	for (size_t i=0; i<entity.size(); ++i) {
		Entity* e = entity[i];
		if (e==NULL) continue;
		for (int w=0; w<para->num_walks; ++w) {
			Behavior* b = e->sampleBehaviorUniformly();
			if (b==NULL) continue;
			vector<int> sentence;
			sentence.push_back(b->entity()->getId());
			sentence.push_back(b->action()->getId());
			for (int l=1; l<para->walk_length; ++l) {
				//logger.info("Now at %d %d", b->entity()->getId(), b->action()->getId());
				b = b->walk(para);
				if (b==NULL) break;
				sentence.push_back(b->entity()->getId());
				sentence.push_back(b->action()->getId());
			}
			corpus.push_back(sentence);
		}
		if (i % 100==0) printf("%d\n", i);
	}
	for (size_t i=0; i<action.size(); ++i) {
		Action* a = action[i];
		if (a==NULL) continue;
		for (int w=0; w<para->num_walks; ++w) {
			Behavior* b = a->sampleBehaviorUniformly();
			if (b==NULL) continue;
			vector<int> sentence;
			sentence.push_back(b->entity()->getId());
			sentence.push_back(b->action()->getId());
			for (int l=1; l<para->walk_length; ++l) {
				//logger.info("Now at %d %d", b->entity()->getId(), b->action()->getId());
				b = b->walk(para);
				if (b==NULL) break;
				sentence.push_back(b->entity()->getId());
				sentence.push_back(b->action()->getId());
			}
			corpus.push_back(sentence);
		}
		if (i % 100==0) printf("%d\n", i);
	}
}

void Graph::rw_parallel(vector<vector<int> >& corpus, Parameter* para) const
{
	vector<vector<int> > local_corpus_entity;
	vector<vector<int> > local_corpus_action;
	omp_set_num_threads(para->num_threads);
	/*
	#pragma omp parallel default(shared) private(local_corpus)
	{
		#pragma omp for schedule(dynamic, 1)
		for (size_t i=0; i<behavior.size(); ++i) {
			int tid = omp_get_thread_num();
			//printf("tid: %d", tid);
			for (int w=0; w<para->num_walks; ++w) {
				vector<int> sentence;
				Behavior* b = behavior[i];
				sentence.push_back(b->entity()->getId());
				sentence.push_back(b->action()->getId());
				for (int l=1; l<para->walk_length; ++l) {
					b = b->walk(para);
					if (b==NULL) break;
					sentence.push_back(b->entity()->getId());
					sentence.push_back(b->action()->getId());
				}
				local_corpus.push_back(sentence);
			}
		}
		#pragma omp critical
		{
			corpus.insert(corpus.end(), local_corpus.begin(), local_corpus.end());
		}
	}
	*/
	#pragma omp parallel default(shared) private(local_corpus_entity)
	{
		#pragma omp for schedule(dynamic, 1)
		for (size_t i=0; i<entity.size(); ++i) {
			Entity* e = entity[i];
			if (e==NULL) continue;
			for (int w=0; w<para->num_walks; ++w) {
				Behavior* b = e->sampleBehaviorUniformly();
				if (b==NULL) continue;
				vector<int> sentence;
				sentence.push_back(b->entity()->getId());
				sentence.push_back(b->action()->getId());
				for (int l=1; l<para->walk_length; ++l) {
					//logger.info("Now at %d %d", b->entity()->getId(), b->action()->getId());
					b = b->walk(para);
					if (b==NULL) break;
					sentence.push_back(b->entity()->getId());
					sentence.push_back(b->action()->getId());
				}
				local_corpus_entity.push_back(sentence);
			}
			if (i % 100==0) printf("entity %d\n", i);
		}
		#pragma omp critical
		{
			corpus.insert(corpus.end(), local_corpus_entity.begin(), local_corpus_entity.end());
		}
	}
	#pragma omp parallel default(shared) private(local_corpus_action)
	{
		#pragma omp for schedule(dynamic, 1)
		for (size_t i=0; i<action.size(); ++i) {
			Action* a = action[i];
			if (a==NULL) continue;
			for (int w=0; w<para->num_walks; ++w) {
				Behavior* b = a->sampleBehaviorUniformly();
				if (b==NULL) continue;
				vector<int> sentence;
				sentence.push_back(b->entity()->getId());
				sentence.push_back(b->action()->getId());
				for (int l=1; l<para->walk_length; ++l) {
					//logger.info("Now at %d %d", b->entity()->getId(), b->action()->getId());
					b = b->walk(para);
					if (b==NULL) break;
					sentence.push_back(b->entity()->getId());
					sentence.push_back(b->action()->getId());
				}
				local_corpus_action.push_back(sentence);
			}
			if (i % 100==0) printf("action %d\n", i);
		}
		#pragma omp critical
		{
			corpus.insert(corpus.end(), local_corpus_action.begin(), local_corpus_action.end());
		}
	}

}

void Graph::preprocess(Parameter* para) {
	log4cpp::Category& root = log4cpp::Category::getRoot();
	log4cpp::Category& logger = root.getInstance("logger");
	logger.info("Preprocess entity ..");
	for (size_t i=0; i<entity.size(); ++i)
	if (entity[i]) {
		//logger.info("%d", i);
		entity[i]->preprocess(para->entity_time_const);
	}
	logger.info("Preprocess action ..");
	for (size_t i=0; i<action.size(); ++i)
	if (action[i])
		action[i]->preprocess(para->action_time_const);
	logger.info("Preprocess down ..");
}


