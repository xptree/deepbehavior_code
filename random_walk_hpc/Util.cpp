#include "Util.h"
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include "Behavior.h"

using namespace std;

vector<string> Util::split(string str, char delim, int max_split)
{
	stringstream ss(str);
	vector<string> elems;
	for (string item; max_split-- && getline(ss, item, delim);elems.push_back(item));
	return elems;
}

vector<long long> Util::string_to_ll(vector<string>& elems)
{
	vector<long long> elems_ll;
	for (size_t i=0; i<elems.size();++i)
		elems_ll.push_back(atoll(elems[i].c_str()));
	return elems_ll;
}

bool Util::cmpBehavior(Behavior* first, Behavior* second)
{
	return first->time() < second->time();
}

long long Util::getFileSize(const char* fileName) // path to file
{
	streampos begin,end;
	ifstream file (fileName, ios::binary);
	begin = file.tellg();
	file.seekg (0, ios::end);
	end = file.tellg();
	file.close();
	return (long long)(end-begin);
}

int Util::firstIndexgeq(const vector<pair<int, int> >& vec, int val)
{
	int l=0, r=int(vec.size())-1, res=vec.size(), mid;
	for (;l<=r;) {
		mid = l+r>>1;
		if (vec[mid].first >= val) {
			res = mid;
			r = mid-1;
		} else {
			l = mid+1;
		}
	}
	return res;
}

int Util::lastIndexleq(const vector<pair<int, int> >& vec, int val)
{
	int l=0, r=int(vec.size())-1, res=-1, mid;
	for (;l<=r;) {
		mid = l+r>>1;
		if (vec[mid].first <= val) {
			res = mid;
			l = mid+1;
		} else {
			r = mid-1;
		}
	}
	return res;
}

double Util::rnd() {
	return (double)rand() / RAND_MAX;
}

