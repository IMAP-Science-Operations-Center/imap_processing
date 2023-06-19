classdef ConsoleHandler < logging.Handler
  
  % Matthew Spaethe
  % Ideas borrowed from Java logging, morphed into Matlab.
  
  methods
    function obj = ConsoleHandler()
    end
  end
  
  methods
    function [] = emit(obj, level, name, varargin)
      %emit  Used to pass log message to Handler from Logger.

      import logging.*
      
      % only print text if local 'logger level' equals or exceeds 'handler level'
      if level >= obj.level
        fprintf(1, '%s: ', Level.getName( level ));
        fprintf(1, '%s: ', name);
        fprintf(1, varargin{:}{:});
        fprintf(1, '\n');
      end
    end
    
    function [] = close(obj)
    end
    
    function [] = flush(obj)
    end
        
  end
  
end

