#pragma once
#include <map>
#include <vector>
using namespace std;

class Entity;
class Action;
class Behavior;
class Parameter;

class Graph
{
private:
	vector<Entity*> entity;
	vector<Action*> action;
	vector<Behavior*> behavior;
public:
	Graph();
	~Graph();
	void addEntity(int);
	void addAction(int);
	void addEntityRelationship(int, int, double);
	void addActionRelationship(int, int, double);
	void addBehavior(int, int, int);
	void rw_serial(vector<vector<int> >& corpus, Parameter*) const;
	void rw_parallel(vector<vector<int> >& corpus, Parameter*) const;
	void preprocess(Parameter*);
};
