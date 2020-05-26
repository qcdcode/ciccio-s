#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#define EXTERN_LOGGER
 #include "logger.hpp"

#include <unistd.h>
#include <sys/ioctl.h>

namespace ciccios
{
  void printBanner()
  {
    /// Width of the message
    const int messageWidth=69;
    
    // Window size
    struct winsize w;
    ioctl(STDOUT_FILENO,TIOCGWINSZ,&w);
    
    /// Check if we are on a terminal
    const bool isTerminal=isatty(STDOUT_FILENO);
    
    /// Terminal output size
    const int width=isTerminal?(w.ws_col):(messageWidth+10);
    
    if(width>=messageWidth)
      {
	/// Padding size
	const int n=(width-messageWidth)/2;
	char sp[n+1];
	for(int i=0;i<n;i++) sp[i]=' ';
	sp[n]='\0';
	
	logger()<<sp<<" _______ _________ _______  _______ _________ _______  _  _______ "<<sp<<endl;
	logger()<<sp<<"(  ____ \\\\__   __/(  ____ \\(  ____ \\\\__   __/(  ___  )( )(  ____ \\"<<sp<<endl;
	logger()<<sp<<"| (    \\/   ) (   | (    \\/| (    \\/   ) (   | (   ) ||/ | (    \\/"<<sp<<endl;
	logger()<<sp<<"| |         | |   | |      | |         | |   | |   | |   | (_____ "<<sp<<endl;
	logger()<<sp<<"| |         | |   | |      | |         | |   | |   | |   (_____  )"<<sp<<endl;
	logger()<<sp<<"| |         | |   | |      | |         | |   | |   | |         ) |"<<sp<<endl;
	logger()<<sp<<"| (____/\\___) (___| (____/\\| (____/\\___) (___| (___) |   /\\____) |"<<sp<<endl;
	logger()<<sp<<"(_______/\\_______/(_______/(_______/\\_______/(_______)   \\_______)"<<sp<<endl;
      }
  };
}
