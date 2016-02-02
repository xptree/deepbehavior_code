#pragma once
#include <vector>
#include <map>
using namespace std;

class Behavior;
class History {
public:
	History();
	~History();
	void preprocess(double time_const);
	pair<double, double> getWeight(int t) const;
	Behavior* sampleBehavior(int t, double time_const) const;
	void addBehavior(Behavior* b);
private:
	vector<Behavior*> h;
	vector<double> prefix, suffix;
	map<double, Behavior*> prefix_cumulative;
	map<double, Behavior*> suffix_cumulative;
	int firstIndexgeq(int t) const;
};
