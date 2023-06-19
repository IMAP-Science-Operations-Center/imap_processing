classdef List < handle
  %List  List based on cell array.
  
  % Matthew Spaethe
  % Ideas borrowed from Java logging, morphed into Matlab.
  
   properties (Access=private)
      list
   end
   
   methods 
      function obj = List()
         obj.list = {};
      end
   end
   
   methods
      function [] = add(obj, x)
        %add  Add item to List.
        obj.list = [obj.list {x}];
      end
      
      function [val] = getList(obj)
        %getList  Return cell array.
        val = obj.list;
      end
      
      function [] = remove(obj, x)
        %remove  Remove item from List.
        
        % TODO: cleaner method of removing handle from cell array?
        for i=1:length(obj.list)
          if x == obj.list{i}
            obj.removeIndex(i);
            break
          end
        end
      end
      
      function [] = removeIndex(obj, index)
        %removeIndex  Remove item at specific position from List.
        obj.list(index) = [];
      end
   end
   
end