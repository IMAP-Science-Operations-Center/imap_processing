classdef Logger < handle
  %Logger  Used for logging messages.
  % Always call Logger.getLogger() to create a Logger or return an existing Logger.
  % Calling Logger.getLogger() without an argument will instruct the method to use the call stack to determine the string name
  % for the Logger.
  % Calling Logger.getLogger(name) will use the provided name to create or return an existing Logger.
  % Calling Logger.getLogger('') with an empty char array will return the root Logger.
  %
  % There are only two levels to the Logger hierarchy.  The root logger is created by default, and is always the parent to any
  % new logger instances.
  %
  % By default, a Logger instance will also use its parent's handlers (useParentHandlers).  The easiest way to visualize this 
  % is to call the method printLoggers of LogManager.  There may be times you want to disable this behavior.  For example, you
  % have a communications library that will log to a separate log file.  You can disable useParentHandlers and add a FileHandler
  % to your communications library logger.  Then log messages will be handled by only your logger's handlers.
  %
  % Matthew Spaethe
  % Ideas borrowed from Java logging, morphed into Matlab.
  
  properties
    handlerList
    level
    name
    parent
    useParentHandlers
    useStackCallName
  end
  
  methods (Access = private)
    function obj = Logger()
      import logging.*
      
      obj.handlerList = List();
      obj.level = LogManager.LOGGER_LOGGING_LEVEL;
      obj.useParentHandlers = LogManager.USE_PARENT_HANDLERS;
      obj.useStackCallName = LogManager.USE_STACK_CALL_NAME;
    end
  end

  methods (Static)
    function logger = getLogger( varargin )
      %getLogger  Always use this static method to create or return an existing Logger instance.
      % Calling Logger.getLogger() without an argument will instruct the method to use the call stack to determine the string name
      % for the Logger.
      % Calling Logger.getLogger(name) will use the provided name to create or return an existing Logger.
      % Calling Logger.getLogger('') with an empty char array will return the root Logger.
      % Java: Find or create a logger for a named subsystem.

      % access global LogManager instance
      logManager = logging.LogManager.getLogManager();
      
      if nargin == 0
        % no name provided, generate name
        st = dbstack(1);
        
        if isempty(st)
          name = 'MatlabCommandWindow';
        else
          for i = length(st):-1:1
            if i == length(st)
              name = st(i).name;
            else 
              name = [name '>' st(i).name];
            end
          end
        end
        
      else
        % class name (or similar) given
        name = varargin{1};
      end
      
      logger = logManager.getLogger( name );
      
      if isempty( logger )
        logger = logging.Logger();
        logger.setName( name );
        
        % set handle to parent logger
        if ~strcmp(name, '')
          logger.setParent( logManager.getLogger('') )
        else
          logger.setParent( [] );
        end
        
        % register new Logger instance with LogManager
        logManager.addLogger( logger );
      end
      
    end
  end
   
  methods
    function addHandler(obj, handler)
      %addHandler  Add handler instance to this logger.
      obj.handlerList.add( handler );
    end
    
    function setLevel(obj, level)
      %setLevel  Set logging level.
      % A log message with a logging level greater than or equal to the logger's logging level will be passed to handlers.
      obj.level = level;
    end
    
    function setName(obj, name)
      %setName  Set the logger's registered name.
      obj.name = name;
    end
    
    function setParent(obj, parent)
      %setParent  Set handle to the logger's parent.
      obj.parent = parent;
    end
    
    function setUseParentHandlers(obj, useParentHandlers)
      %setUseParentHandlers  Set useParentHandlers flag.
      % If true, the parent's handlers will be used in addition to the logger's handlers.  That is, log messages will
      % be forwarded to the logger's handlers and the parent's handlers if the logging level has been met.
      obj.useParentHandlers = useParentHandlers;
    end
    
    function setUseStackCallName(obj, useStackCallName)
      %setUseStackCallName  Set useStackCallName flag.
      % When passing log messages to the handlers, this flag determines if the call stack (at point of log call) or the
      % registered name should be used.
      obj.useStackCallName = useStackCallName;
    end
    
    function [parent] = getParent(obj)
      %getParent  Returns handle to parent Logger.
      parent = obj.parent;
    end
    
    function [handlers] = getHandlers(obj, varargin)
      %getHandlers  Returns a cell array of handles to Handler objects.
      % getHandlers() returns a cell array of handles to all Handler objects for this Logger.
      % getHandlers(classname) returns a cell array of handles to all Handler objects for this Logger of type classname.
      handlers = [];
      
      if nargin == 1
        handlers = obj.handlerList.getList();
      else
        for i = obj.handlerList.getList()
          if isa(i{1}, varargin{1})
            handlers = [handlers i];
          end
        end
      end
    end
            
    function [level] = getLevel(obj)
      %getLevel  Return the logging level for this Logger instance.
      % A log message with a logging level greater than or equal to the logger's logging level will be passed to handlers.
      level = obj.level;
    end
    
    function [name] = getName(obj)
      %getName  Return the logger's registered name.
      name = obj.name;
    end
    
    function [useParentHandlers] = getUseParentHandlers(obj)
      %getUseParentHandlers  Return the useParentHandlers flag.
      % If true, the parent's handlers will be used in addition to the logger's handlers.  That is, log messages will
      % be forwarded to the logger's handlers and the parent's handlers if the logging level has been met.
      useParentHandlers = obj.useParentHandlers;
    end
    
    function [useStackCallName] = getUseStackCallName(obj)
      %getUseStackCallName  Return the useStackCallName flag.
      % When passing log messages to the handlers, this flag determines if the call stack (at point of log call) or the
      % registered name should be used.
      useStackCallName = obj.useStackCallName;
    end
    
    function [] = reset(obj)
      %reset  Close and remove handlers.
      % Instruct registered handlers to close().
      % Remove any registered handlers that are not "sticky" (e.g. root console handler).      
      for handler = obj.handlerList.getList()
        if ~handler{1}.getSticky()
          obj.handlerList.remove( handler{1} );
          handler{1}.close();
          delete( handler{1} );
        end
      end
    end
    
    
    function log(obj, level, varargin)
      %log  Generate log message of specified logging level.
      % Example usage:
      %
      %   import logging.*
      %   log(Level.INFO, 'Received %d packets', receivedPacketCount);
      %
      %   log(logging.Level.INFO, 'Received %d packets', receivedPacketCount);
      obj.logm( 2, level, varargin{:} );
    end
    
    function error(obj, varargin)
      %error  Convenience method for logging message with level ERROR.
      obj.logm( 2, logging.Level.ERROR, varargin{:} );
    end
    
    function warning(obj, varargin)
      %warning  Convenience method for logging message with level WARNING.
      obj.logm( 2, logging.Level.WARNING, varargin{:} );
    end
    
    function info(obj, varargin)
      %info  Convenience method for logging message with level INFO.
      obj.logm( 2, logging.Level.INFO, varargin{:} );
    end
    
    function fine(obj, varargin)
      %fine  Convenience method for logging message with level FINE.
      obj.logm( 2, logging.Level.FINE, varargin{:} );
    end
    
    function finer(obj, varargin)
      %finer  Convenience method for logging message with level FINER.
      obj.logm( 2, logging.Level.FINER, varargin{:} );
    end
    
    function finest(obj, varargin)
      %finest  Convenience method for logging message with level FINEST.
      obj.logm( 2, logging.Level.FINEST, varargin{:} );
    end
  end
  
  methods (Access = private)
    function logm(obj, stackPtr, level, varargin)
      %logm  Internal method used to log messages.
      if level >= obj.level

        if obj.useStackCallName
          st = dbstack(stackPtr);

          if isempty(st)
            stackCallName = 'MatlabCommandWindow';
          else
            for i = length(st):-1:1
              if i == length(st)
                stackCallName = st(i).name;
              else 
                stackCallName = [stackCallName '>' st(i).name];
              end
            end
          end
        end
          
        for handler = obj.handlerList.getList()
          % stackCallName results in similar log message to Java logp()
          if obj.useStackCallName
            handler{1}.emit( level, stackCallName, varargin );
          else
            handler{1}.emit( level, obj.name, varargin );
          end
        end
        
        % --- use parent handlers ---
        
        if obj.useParentHandlers
          logger = obj.parent;
        else
          logger = [];
        end

        % root logger will have empty parent handle
        while ~isempty( logger )
          for handler = logger.handlerList.getList()
            if obj.useStackCallName
              handler{1}.emit( level, stackCallName, varargin );
            else
              handler{1}.emit( level, obj.name, varargin );
            end
          end
          
          % continue up hierarchy
          if logger.useParentHandlers
            logger = logger.parent;
          else
            logger = [];
          end
        end
        
        % -----------------------
        
      end
      
%       % --- use parent loggers ---      
%
%       % propagate logged message up the chain?
%       if obj.useParentHandlers
%         logger = obj.parent;
%         % root logger will have empty parent handle
%         if ~isempty( logger )
%           logger.logm( stackPtr + 1, level, varargin{:} );
%         end
%       end
%
%       % ----------------------
      
    end
  end
  
end

