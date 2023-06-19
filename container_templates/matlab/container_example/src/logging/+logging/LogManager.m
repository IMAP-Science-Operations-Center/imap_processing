classdef LogManager < logging.Singleton
  %LogManager  Global LogManager instance
  % There is a global LogManager object (Singleton).
  % Call static method LogManager.getLogManager() to return the global LogManager object.
  %
  % Matthew Spaethe
  % Ideas borrowed from Java logging, morphed into Matlab.
  
  properties (Constant)
    ROOT_LOGGER_LOGGING_LEVEL = logging.Level.INFO;
    ROOT_CONSOLE_HANDLER_LOGGING_LEVEL = logging.Level.ALL;
    
    LOGGER_LOGGING_LEVEL = logging.Level.INFO;
    USE_PARENT_HANDLERS  = true;
    USE_STACK_CALL_NAME  = true;
  end

  properties
    loggerMap
  end
  
  methods (Access = private)
    function obj = LogManager()
      obj.loggerMap = containers.Map();
    end
  end
    
  methods (Static)
    function obj = instance()
      persistent singleInstance
      
      if isempty( singleInstance )
        obj = logging.LogManager();
        singleInstance = obj;

        % create root logger and its console handler
        obj.createRootLogger();
      else
        obj = singleInstance;
      end
           
    end
    
    function obj = getLogManager()
      %getLogManager  Return the global LogManager object.
      % Wrapper to make method call similar to Java.
      obj = logging.LogManager.instance();
    end
  end
  
  methods (Access = private)
    function [] = createRootLogger(obj)
      %createRootLogger  Add the default root logger and its console handler.
      % We set the 'sticky' attribute so the console handler is not deleted by reset().
      import logging.*
      
      rootConsoleHandler = ConsoleHandler();
      rootConsoleHandler.setLevel( LogManager.ROOT_CONSOLE_HANDLER_LOGGING_LEVEL );
      rootConsoleHandler.setSticky( true );
      
      rootLogger = Logger.getLogger('');
      rootLogger.setLevel( LogManager.ROOT_LOGGER_LOGGING_LEVEL );
      rootLogger.addHandler( rootConsoleHandler );
    end
  end
   
  methods
    
    function delete(obj)
    end
    
    function [logger] = getLogger(obj, name)
      %getLogger  Returns handle to named Logger, or empty array if not found.
      % Java: matching logger or null if none is found
      % 
      if ~obj.loggerMap.isKey( name )
        logger = [];
      else
        logger = obj.loggerMap( name );
      end
    end
    
    function [success] = addLogger(obj, logger)
      %addLogger  Logger class will call this method to add a new Logger object to LogManager.
      % Java:  Add a named logger.  This does nothing and returns false if a logger with the same name is already registered. 
      if ~obj.loggerMap.isKey( logger.getName() )
        % add to map
        obj.loggerMap( logger.getName() ) = logger;
        success = true;
      else
        success = false;
      end
    end
    
    function [] = reset(obj)
      %reset  Close and remove handlers.
      % Each logger will instruct its registerd handlers to close().
      % Each logger will remove any registered handlers that are not "sticky" (e.g. root console handler).
      import logging.*
      
      keys = obj.loggerMap.keys();
      for key = keys
        logger = obj.loggerMap( key{1} );
        logger.reset();
      end
    end
    
    function [] = resetAll(obj)
      %resetAll  Close and remove handlers, remove loggers, reset root logger and root console handler logging levels.
      % Logger reset:
      %   Each logger will instruct its associated handlers to close().
      %   Each logger will remove any handlers that are not "sticky" (e.g. root console handler).
      % Delete loggers (except the root logger).
      % Set the root logger back to its default logging level.
      % Set the root console handler back to its default logging level.
      import logging.*
      
      keys = obj.loggerMap.keys();
      for key = keys
        logger = obj.loggerMap( key{1} );
        logger.reset();
        if ~strcmp(logger.name, '')
          %delete( obj.loggerMap(key{1}) );
          obj.loggerMap.remove( key{1} );
        else
          logger.setLevel( LogManager.ROOT_LOGGER_LOGGING_LEVEL );
          % get handle to the root console handler and restore its default logging level
          consoleHandlers = logger.getHandlers('logging.ConsoleHandler');
          consoleHandlers{1}.setLevel( LogManager.ROOT_CONSOLE_HANDLER_LOGGING_LEVEL );
        end
      end
    end
    
    function [keys] = getLoggerNames(obj)
      %getLoggerNames  Returns cell array of Logger registered names.
      keys = obj.loggerMap.keys();
    end
    
    function [] = printLoggers(obj)
      %printLoggers  Prints table of loggers, associated handlers, and 'use parent handler flag' (Y/N).
      fprintf('\n');
      fprintf('--- LogManager table ---\n');
      
      loggerNames = obj.loggerMap.keys();
      for loggerName = loggerNames
        logger = obj.loggerMap( loggerName{1} );
        if strcmp( logger.getName(), '' )
          indent = 0;
        else
          indent = 2;
        end
        
        for i = 1:indent
          fprintf(' ');
        end
        
        switch logger.getUseParentHandlers()
          case 0, useParentLogger = 'N';
          case 1, useParentLogger = 'Y';
        end
          
        fprintf( '+ LOGGER ''%s'' %s [%s]\n', logger.getName(), logging.Level.getName(logger.getLevel()), useParentLogger );

        handlers = logger.getHandlers();
        for handler = handlers
          for i = 1:indent
            fprintf(' ');
          end
          fprintf( '    HANDLER %s %s\n', class(handler{1}), logging.Level.getName(handler{1}.getLevel()) );
        end
      end
      fprintf('\n');
    end
    
  end
  
end