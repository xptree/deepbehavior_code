#pragma once
#include <vector>

class Behavior;
class History;

using namespace std;

class Action
{
private:
	int id;
	vector<pair<Action*, double> > neighbor;
	History* history;
public:
	Action(int);
	~Action();
	void addActionRelationship(Action*, double);
	void addBehavior(Behavior*);
	//void rw_serial(vector<vector<int> >& corpus, int max_len) const;
	Action* sampleActionFromNeighbor(int t, double time_const, double relation_const) const;
	Behavior* sampleBehaviorFromHistory(int t, double time_const) const;
	double getWeight(int t, double time_const) const;
	void preprocess(double time_const);
	int getId() const;
};
