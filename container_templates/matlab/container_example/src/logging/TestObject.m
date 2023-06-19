classdef TestObject < handle
  
  % Matthew Spaethe
  % Ideas borrowed from Java logging, morphed into Matlab.
    
  properties (Access = private)
    logger
  end
  
  methods
    
    function obj = TestObject()
      import logging.*
      
      % use class name when registering this logger
      % multiple instances of this class will use the same logger instance since we are registering via class name
      obj.logger = Logger.getLogger( class(obj) );
      obj.logger.setLevel( Level.INFO );
      obj.logger.log(Level.FINEST, 'constructor');
    end
    
    function [] = run(obj)
      obj.logger.log(logging.Level.INFO, 'example printf type %s', 'call');
      
      % blah blah
      % blah blah 
      
      obj.logger.info('using convenience method for logging');
      
      % blah blah
      % blah blah 
      
      obj.logger.setUseStackCallName( false );
      obj.logger.warning('disabled stack call name generation; use registered name in log messages');

    end
      
  end
end
