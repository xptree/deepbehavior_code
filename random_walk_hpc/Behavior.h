#pragma once
using namespace std;

class Action;
class Entity;
class Parameter;

class Behavior {
public:
	Behavior(Entity* e, Action* a, int t);
	~Behavior();
	Entity* entity() const;
	Action* action() const;
	int time() const;
	Behavior* walk(Parameter*) const;
private:
	Entity* e;
	Action* a;
	int t;
};

