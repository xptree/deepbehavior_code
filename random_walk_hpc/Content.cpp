#include "Content.h"
#include "Util.h"
#include <cstdlib>
#include <cstring>

Content::Content(string str)
{
	vector<string> token = Util::split(str, '\t');
	for (size_t i=0; i<token.size(); ++i)
		action.push_back(atoi(token[i].c_str()));
}


Content::~Content() {}

int Content::getAction() const
{
	int pos = rand() % action.size();
	return action[pos];
}
