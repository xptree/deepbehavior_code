#pragma once
#include <vector>
#include <string>

using namespace std;

class Content
{
private:
	vector<int> action;
public:
	Content(string);
	~Content();
	int getAction() const;
};
