% Matthew Spaethe
% Ideas borrowed from Java logging, morphed into Matlab.

import logging.*


% --- BASIC CONSOLE ---

% get the global LogManager
logManager = LogManager.getLogManager();

% add a logger instance to this script
logger = Logger.getLogger('TestScript');
logger.setLevel( Level.ALL );

logger.info('Hi, this is info!');
logger.warning('Hi, this is warning!');

% side-effect is to close all handlers
logManager.resetAll();


% --- BASIC CONSOLE WITH FILE ---

% get the global LogManager
logManager = LogManager.getLogManager();

% add a file handler to the root logger
fileHandler = FileHandler('./Basic-RootFileHandler.log');
fileHandler.setLevel( Level.ALL );
rootLogger = logManager.getLogger('');
rootLogger.addHandler( fileHandler );

% add a logger instance to this script
% will use stack to generate name for logger since a name is not being provided
logger = Logger.getLogger( );
logger.setLevel( Level.ALL );

logger.info('Hi, this is info!');
logger.warning('Hi, this is warning!');

% side-effect is to close all handlers
logManager.resetAll();


% --- A LITTLE BEYOND ---

% get the global LogManager
logManager = LogManager.getLogManager();

% in case LogManager exists from a previous run
% removes all handlers -- except for the root console handler
% removes all loggers  -- except for the root logger
% logManager.resetAll();

% removes all handlers -- except for the root console handler
% logManager.reset();

% obtain handle to root logger and set its logging level
% output log messages of level INFO and above
rootLogger = logManager.getLogger('');
rootLogger.setLevel( Level.INFO );

% add a file handler to the root logger
fileHandler = FileHandler('./RootFileHandler.log');
fileHandler.setLevel( Level.ALL );
rootLogger.addHandler( fileHandler );

% add a logger instance to this script
% will use stack to generate name for logger since a name is not being provided
logger = Logger.getLogger( );
logger.setLevel( Level.ALL );

% add a file handler to this script's logger
fileHandler = FileHandler('./ScriptFileHandler.log');
fileHandler.setLevel( Level.ALL );
logger.addHandler( fileHandler );

logger.info('hi');

% don't propagate log messages to the parent's handlers
logger.setUseParentHandlers( false );
logger.log( Level.WARNING, 'yo' );

% create an instance of TestObject
testObject = TestObject();
testObject.run();

% create an instance of TestObject2
testObject2 = TestObject2();
testObject2.run();

% display table of all registered loggers and their associated handlers
logManager = LogManager.getLogManager();
logManager.printLoggers();

% change the root console handler's logging level
logManager      = LogManager.getLogManager();
logger          = logManager.getLogger('');
consoleHandlers = logger.getHandlers('logging.ConsoleHandler');
consoleHandlers{1}.setLevel( Level.INFO );

% side-effect is to close all handlers (e.g. file handler)
% removes all handlers -- except for the root console handler
logManager.reset();
logManager.printLoggers();

% removes all loggers and handlers -- except for the root logger and the root console handler
logManager.resetAll();
logManager.printLoggers();