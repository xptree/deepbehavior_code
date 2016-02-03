#include "History.h"
#include "Behavior.h"
#include "Util.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <cstdlib>
#include "log4cpp/Category.hh"
#include "log4cpp/OstreamAppender.hh"
#include "log4cpp/FileAppender.hh"
#include "log4cpp/Priority.hh"
#include "log4cpp/PatternLayout.hh"

History::History() {}
History::~History() {}

void History::addBehavior(Behavior* b) {
	h.push_back(b);
}


Behavior* History::sampleBehaviorUniformly() const {
	if (h.empty()) return NULL;
	return h[rand() % h.size()];
}

void History::preprocess(double time_const) {
	/*
	log4cpp::Category& root = log4cpp::Category::getRoot();
	log4cpp::Category& logger = root.getInstance("logger");
	logger.info("history length = %d time_const = %.2lf", h.size(), time_const);
	*/
	if (h.empty()) return;
	sort(h.begin(), h.end(), Util::cmpBehavior);
	//for (size_t i=0; i<h.size(); ++i) logger.info("time = %d", h[i]->time());
	prefix.resize(h.size(), 0.0);
	suffix.resize(h.size(), 0.0);
	double sum = 0.0;
	for (int i=0; i<h.size(); ++i) {
		sum += exp(h[i]->time() / time_const);
		//logger.info("%d %.2lf",i, sum);
		prefix[i] = sum;
		prefix_cumulative[sum] = h[i];
	}
	sum = 0.0;
	for (int i=h.size()-1; i>=0; --i) {
		sum += exp(-h[i]->time() / time_const);
		//logger.info("%d %.2lf",i, sum);
		suffix[i] = sum;
		suffix_cumulative[sum] = h[i];
	}
}

pair<double, double> History::getWeight(int t) const {
	if (h.empty()) return make_pair(0.0, 0.0);
	int res = firstIndexgeq(t);
	if (res == 0) return make_pair(0.0, suffix[res]);
	if (res == h.size()) return make_pair(prefix[res-1], 0);
	return make_pair(prefix[res-1], suffix[res]);
}


int History::firstIndexgeq(int t) const {
	int l=0, r=int(h.size())-1, res=h.size(), mid;
	for (;l<=r;) {
		mid = l+r>>1;
		if (h[mid]->time() >= t) {
			res = mid;
			r = mid-1;
		} else {
			l = mid+1;
		}
	}
	return res;
}

Behavior* History::sampleBehavior(int t, double time_const) const {
	if (h.empty()) return NULL;
	int res = firstIndexgeq(t);
	bool left = false;
	if (res == 0) { left = false; }
	else if (res == h.size()) { left = true; }
	else {
		double lweight = exp(-t/time_const) * prefix[res-1];
		double rweight = exp(t/time_const) * suffix[res];
		left = Util::rnd() * (lweight + rweight) <= lweight;
	}
	if (left) {
		double linear = Util::rnd() * prefix[res-1];
		return prefix_cumulative.upper_bound(linear)->second;
	} else {
		double linear = Util::rnd() * suffix[res];
		return suffix_cumulative.upper_bound(linear)->second;
	}
	return NULL;
}
