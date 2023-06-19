classdef FileHandler < logging.Handler

  % Matthew Spaethe
  % Ideas borrowed from Java logging, morphed into Matlab.

  properties
    fileId
    fileOpen = false;
    filename
    logger
  end
  
  methods
    function obj = FileHandler(filename)
      %FileHandler  Handler subclass used to log messages to file.
      %  FileHandler(filename)  Log output to file, creating / overwriting filename.
      
      import logging.*

      obj.logger = Logger.getLogger( );
      obj.logger.setLevel( Level.FINEST );
      obj.logger.log(Level.FINEST, 'constructor');

      obj.logger.fine('opening file: %s', filename);
      obj.fileId = fopen(filename, 'w');
      
      if obj.fileId == -1
        obj.logger.log(Level.ERROR, 'error in opening file %s', filename);
        obj.fileOpen = false;
        obj.filename = [];
      else
        obj.fileOpen = true;
        obj.filename = filename;
      end
        
    end
    
    function delete(obj)
    end
    
    function emit(obj, level, name, varargin)
      %emit  Used to pass log message to Handler from Logger.
      
      import logging.*
      
      % only print text if logged message level equals or exceeds 'handler level'
      if level >= obj.level
        fprintf(obj.fileId, '%s: ', Level.getName( level ));
        fprintf(obj.fileId, '%s: ', name);
        fprintf(obj.fileId, varargin{:}{:});
        fprintf(obj.fileId, '\n');
      end
    end
    
    function [] = close(obj)
      %close  Called by Logger when emit() will no longer be called.  Close any relevant i/o.
      if obj.fileOpen
        try
          fclose( obj.fileId );
          obj.logger.fine('closed file %s', obj.filename);
          obj.fileOpen = false;
        catch e
          obj.logger.error(e.message);
        end
      end
    end
    
    function [] = flush(obj)
      
    end
        
  end
end

