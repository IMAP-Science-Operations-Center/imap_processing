classdef Handler < handle
  %Handler  Used by Logger to "emit" log messages.
  % Additional handler classes can be written by subclassing Handler.
  
  % Matthew Spaethe
  % Ideas borrowed from Java logging, morphed into Matlab.
  
  properties
    level = logging.Level.OFF;
    sticky = false;
  end
  
  methods
    function setLevel(obj, level)
      %setLevel  Set handler logging level.
      % Log messages presented by the Logger with a logging level greater than or equal to the Handler logging level should be emitted.
      obj.level = level;
    end
    
    function setSticky(obj, sticky)
      %setSticky  Set the sticky flag.
      % If a Handler is marked as sticky, it will not be removed from the Logger instance when the Logger is reset.
      obj.sticky = sticky;
    end
    
    function [level] = getLevel(obj)
      %getLevel  Return the handler logging level.
      % Log messages presented by the Logger with a logging level greater than or equal to the Handler logging level should be emitted.
      level = obj.level;
    end
    
    function [sticky] = getSticky(obj)
      %getSticky  Return the sticky flag.
      % If a Handler is marked as sticky, it will not be removed from the Logger instance when the Logger is reset.
      sticky = obj.sticky;
    end
  end
  
  methods (Abstract)
    %emit  Method used to pass log message to Handler from Logger.
    [] = emit(obj, level, name, varargin);
    %close  Method called by Logger when emit() will no longer be called.  Close any relevant i/o.
    [] = close(obj);
    %flush  Method called by Logger to request buffer to be flushed.
    [] = flush(obj);
  end

end
