import { DropResult } from 'react-beautiful-dnd';
import { Operation } from '@/app/types';
import { Droppable, DragDropContext } from 'react-beautiful-dnd';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { OperationCard } from '@/components/OperationCard';
import { Button } from '@/components/ui/button';
import { Plus, ChevronDown, Play } from 'lucide-react';


const PipelineGUI: React.FC<{ 
    operations: Operation[]; 
    onAddOperation: (llmType: string, type: string, name: string) => void;
    onDragEnd: (result: DropResult) => void;
    onRunAll: () => void;
    onDeleteOperation: (id: string) => void;
    onUpdateOperation: (id: string, updatedOperation: Operation) => void;
  }> = ({ operations, onAddOperation, onDragEnd, onRunAll, onDeleteOperation, onUpdateOperation }) => (
    <div className="h-full p-4 bg-white overflow-auto">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-sm font-bold uppercase">Pipeline</h2>
        <div className="flex space-x-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button size="sm" className="rounded-sm">
                <Plus size={16} className="mr-2" /> Add Operation <ChevronDown size={16} className="ml-2" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent>
              <DropdownMenuLabel>LLM Operations</DropdownMenuLabel>
              <DropdownMenuItem onClick={() => onAddOperation('LLM', 'Map', 'Untitled Map')}>Map</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onAddOperation('LLM', 'Reduce', 'Untitled Reduce')}>Reduce</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onAddOperation('LLM', 'Resolve', 'Untitled Resolve')}>Resolve</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onAddOperation('LLM', 'Equijoin', 'Untitled Equijoin')}>Equijoin</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onAddOperation('LLM', 'Filter', 'Untitled Filter')}>Filter</DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuLabel>Non-LLM Operations</DropdownMenuLabel>
              <DropdownMenuItem onClick={() => onAddOperation('non-LLM', 'Unnest', 'Untitled Unnest')}>Unnest</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onAddOperation('non-LLM', 'Split', 'Untitled Split')}>Split</DropdownMenuItem>
              <DropdownMenuItem onClick={() => onAddOperation('non-LLM', 'Gather', 'Untitled Gather')}>Gather</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <Button onClick={onRunAll} size="sm" className="rounded-sm">
            <Play size={16} className="mr-2" /> Run All
          </Button>
        </div>
      </div>
      <DragDropContext onDragEnd={onDragEnd}>
        <Droppable droppableId="operations">
          {(provided) => (
            <div {...provided.droppableProps} ref={provided.innerRef}>
              {operations.map((op, index) => (
                <OperationCard 
                  key={op.id} 
                  operation={op} 
                  index={index} 
                  onDelete={onDeleteOperation}
                  onUpdate={onUpdateOperation}
                />
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
    </div>
  );

export default PipelineGUI;