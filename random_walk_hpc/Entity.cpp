#include "Entity.h"
#include "Util.h"
#include "History.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <cmath>

Entity::Entity(int id) : id(id), history(new History()) {}

Entity::~Entity() {}

void Entity::addEntityRelationship(Entity* e, double weight)
{
	neighbor.push_back(make_pair(e, weight));
}

int Entity::getId() const {return id;}

void Entity::addBehavior(Behavior* b)
{
	history->addBehavior(b);
}

Behavior* Entity::sampleBehaviorUniformly() const {return history->sampleBehaviorUniformly();}
/*
void Entity::rw_serial(vector<vector<int> >& corpus, int max_len, int delta) const
{
	for (size_t i=0; i<history.size(); ++i) {
		// start from history[i]
		vector<int> sentence;
		sentence.push_back(id);
		sentence.push_back(history[i].second);
		Entity* pnode = (Entity*)this;
		int t       = history[i].first;
		for (int j=1; j<max_len; ++j) {
			pnode = pnode->sampleNextEntity(t, delta);
			int action = pnode->sampleNextAction(t, delta);
			sentence.push_back(pnode->getId());
			sentence.push_back(action);
		}
		corpus.push_back(sentence);
	}
}
*/

Entity* Entity::sampleEntityFromNeighbor(int t, double time_const, double relation_const) const
{
	if (neighbor.empty()) return NULL;
	vector<double> weights;
	for (size_t i=0; i<neighbor.size(); ++i) {
		Entity* e = neighbor[i].first;
		double relation = neighbor[i].second;
		double this_weight = e->getWeight(t, time_const) * exp(-relation_const/relation);
		weights.push_back(this_weight);
	}
	for (size_t i=1; i<weights.size(); ++i)
		weights[i] += weights[i-1];
	double select = Util::rnd() * weights.back();
	for (size_t i=0; i<weights.size(); ++i)
	if (select <= weights[i]) return neighbor[i].first;
	return neighbor.back().first;
}

Behavior* Entity::sampleBehaviorFromHistory(int t, double time_const) const {
	return history->sampleBehavior(t, time_const);
}

double Entity::getWeight(int t, double time_const) const
{
	auto weight = history->getWeight(t);
	double res = exp(-t/time_const) * weight.first
		+ exp(t/time_const) * weight.second;
	return res;
}

/*
int Entity::sampleNextAction(int& t, int delta) const
{
	int start = Util::firstIndexgeq(history, t-delta);
	int end = Util::lastIndexleq(history, t+delta);
	assert(end >= start);
	int select = start + rand() % (end - start + 1);
	t = history[select].first;
	return history[select].second;
}
*/

void Entity::preprocess(double time_const)
{
	history->preprocess(time_const);
}


