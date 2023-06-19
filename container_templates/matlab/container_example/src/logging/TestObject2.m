classdef TestObject2 < handle
  
  % Matthew Spaethe
  % Ideas borrowed from Java logging, morphed into Matlab.
    
  properties (Constant)
    logger = logging.Logger.getLogger('TestObject2');
  end
  
  methods
    
    function obj = TestObject2()
      import logging.*
      
      TestObject2.logger.setLevel( Level.INFO );
      TestObject2.logger.log(Level.FINEST, 'constructor');
    end
    
    function [] = run(obj)
      TestObject2.logger.log(logging.Level.INFO, 'example printf type %s', 'call');
      
      % blah blah
      % blah blah 
      
      TestObject2.logger.info('using convenience method for logging');
      
      % blah blah
      % blah blah 
      
      TestObject2.logger.setUseStackCallName( false );
      TestObject2.logger.warning('disabled stack call name generation; use registered name in log messages');

    end
      
  end
end
