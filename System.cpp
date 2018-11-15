#include "System.h"

System::System(const string& strVocFile, const string& strSysParams, const string& strAssoc)
{
  
  SysParams sysparams(strSysParams);
  mpVocabulary = new ORBVocabulary();
  bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
  if(!bVocLoad)return;
  
}
