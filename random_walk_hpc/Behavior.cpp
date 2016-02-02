#include "Behavior.h"
#include "Entity.h"
#include "Action.h"
#include "Parameter.h"
#include "Util.h"
#include "log4cpp/Category.hh"
#include "log4cpp/OstreamAppender.hh"
#include "log4cpp/FileAppender.hh"
#include "log4cpp/Priority.hh"
#include "log4cpp/PatternLayout.hh"


Behavior::Behavior(Entity* e, Action* a, int t) : e(e), a(a), t(t) {}

Behavior::~Behavior() {}

Entity* Behavior::entity() const {return e;}
Action* Behavior::action() const {return a;}
int Behavior::time() const {return t;}

Behavior* Behavior::walk(Parameter* para) const {
	double select = Util::rnd();
	/*
	log4cpp::Category& root = log4cpp::Category::getRoot();
	log4cpp::Category& logger = root.getInstance("logger");
	logger.info("select = %.2lf", select);
	*/
	Behavior* b_next = NULL;
	if (select <= para->self_entity) {
		b_next = e->sampleBehaviorFromHistory(t, para->entity_time_const);
		if (b_next) return b_next;
	}
	if (select <= para->self_entity + para->neighbor_entity) {
		Entity* e_next = e->sampleEntityFromNeighbor(t, para->entity_time_const, para->entity_relation_const);
		if (e_next) b_next = e_next->sampleBehaviorFromHistory(t, para->entity_time_const);
		if (b_next) return b_next;
	}
	if (select <= para->self_entity + para->neighbor_entity + para->self_action) {
		b_next = a->sampleBehaviorFromHistory(t, para->action_time_const);
		if (b_next) return b_next;
	}
	Action* a_next = a->sampleActionFromNeighbor(t, para->action_time_const, para->action_relation_const);
	if (a_next) b_next = a_next->sampleBehaviorFromHistory(t, para->action_time_const);
	return b_next;
}
