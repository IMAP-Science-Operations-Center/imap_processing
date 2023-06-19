classdef Level
  %Level Class containing defined logging levels.
  
  % Matthew Spaethe
  % Ideas borrowed from Java logging, morphed into Matlab.
  
  properties (Constant)
    % Logging level
    %
    % Java
    %  • SEVERE  1000
    %  • WARNING  900
    %  • INFO     800
    %  • CONFIG   700
    %  • FINE     500
    %  • FINER    400
    %  • FINEST   300
    
    % Python
    %  • CRITICAL  50
    %  • ERROR     40
    %  • WARNING   30
    %  • INFO      20
    %  • DEBUG     10
    %  • NOTSET     0
    
    ERROR   =  60;
    WARNING =  50;
    INFO    =  40;
    FINE    =  30;
    FINER   =  20;
    FINEST  =  10;
    
    OFF     = 255;
    ALL     =   0;
  end
  
  methods (Static)
    function [str] = getName(level)
      %getName  Returns string name for numeric logging level.
      
      import logging.*
      
      switch level
        case Level.ERROR,   str = 'ERROR';
        case Level.WARNING, str = 'WARNING';
        case Level.INFO,    str = 'INFO';
        case Level.FINE,    str = 'FINE';
        case Level.FINER,   str = 'FINER';
        case Level.FINEST,  str = 'FINEST';
          
        case Level.OFF,     str = 'OFF';
        case Level.ALL,     str = 'ALL';
          
        otherwise,          str = ['CUSTOM (' num2str(level) ')'];
      end
    end
    
    function level = getLevel(name)
      %getLevel  Returns numeric logging level for corresponding string.
      
      import logging.*
      
      switch name
        case 'ERROR',   level = Level.ERROR;
        case 'WARNING', level = Level.WARNING;
        case 'INFO',    level = Level.INFO;
        case 'FINE',    level = Level.FINE;
        case 'FINER',   level = Level.FINER;
        case 'FINEST',  level = Level.FINEST;
        case 'OFF',     level = Level.OFF;
        case 'ALL',     level = Level.ALL;
        otherwise,      level = -1; 
      end
    end
    
  end
  
end

