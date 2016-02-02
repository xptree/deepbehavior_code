#pragma once
#include <vector>
#include <string>

using namespace std;
class Behavior;
class Util
{
public:
	static vector<string> split(string str, char delim, int max_split=2147483647);
	static vector<long long> string_to_ll(vector<string>& elems);
	static bool cmpBehavior(Behavior*, Behavior*);
	static long long getFileSize(const char* fileName);
	static int firstIndexgeq(const vector<pair<int, int> >&, int val);
	static int lastIndexleq(const vector<pair<int, int> >&, int val);
	static double rnd();
};


