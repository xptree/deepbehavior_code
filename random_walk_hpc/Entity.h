#pragma once
#include <vector>
using namespace std;

class Parameter;
class Behavior;
class History;

class Entity
{
private:
	int id;
	vector<pair<Entity*, double> > neighbor;
	History* history;
public:
	Entity(int);
	~Entity();
	void addEntityRelationship(Entity*, double);
	void addBehavior(Behavior*);
	//void rw_serial(vector<vector<int> >& corpus, int max_len, int delta) const;
	Entity* sampleEntityFromNeighbor(int t, double time_const, double relation_const) const;
	Behavior* sampleBehaviorFromHistory(int t, double time_const) const;
	double getWeight(int t, double time_const) const;
	//void sortHistoryByTime();
	void preprocess(double time_const);
	int getId() const;
};
