#ifndef SYSTEM_H
#define SYSTEM_H
#include <thread>
#include "base.h"
#include "SysParams.h"

class System 
{
public:
  System(const string& strVocFile, const string& strSysParams, const string& strAssoc);
  
  
  //SysParams sysparams;
 
  ORBVocabulary* mpVocabulary;
  
};



#endif
