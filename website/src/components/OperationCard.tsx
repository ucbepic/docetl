import React, { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Card, CardHeader, CardContent } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { DragDropContext, Droppable, Draggable, DropResult } from 'react-beautiful-dnd';
import { FileText, Maximize2, Minimize2, Plus, Play, GripVertical, Trash2, ChevronDown, Zap, Edit2, Settings } from 'lucide-react';
import { Operation, SchemaItem, SchemaType } from '@/app/types';
import { usePipelineContext } from '@/contexts/PipelineContext';

const SchemaForm: React.FC<{
    schema: SchemaItem[];
    onUpdate: (newSchema: SchemaItem[]) => void;
    level?: number;
    isList?: boolean;
  }> = ({ schema, onUpdate, level = 0, isList = false }) => {
    const addItem = () => {
      if (isList) return;
      onUpdate([...schema, { key: '', type: 'string' }]);
    };
  
    const updateItem = (index: number, item: SchemaItem) => {
      const newSchema = [...schema];
      newSchema[index] = item;
      onUpdate(newSchema);
    };
  
    const removeItem = (index: number) => {
      if (isList) return;
      const newSchema = schema.filter((_, i) => i !== index);
      onUpdate(newSchema);
    };
  
    return (
      <div style={{ marginLeft: `${level * 20}px` }}>
        {schema.map((item, index) => (
          <div key={index} className="flex flex-wrap items-center space-x-2 mb-2">
            {!isList && (
              <Input
                value={item.key}
                onChange={(e) => updateItem(index, { ...item, key: e.target.value })}
                placeholder="Key"
                className="w-1/3 min-w-[150px]"
              />
            )}
            <Select
              value={item.type}
              onValueChange={(value: SchemaType) => {
                updateItem(index, {
                  ...item,
                  type: value,
                  subType: value === 'list' ? { key: '0', type: 'string' } :
                           value === 'dict' ? [{ key: '', type: 'string' }] :
                           undefined
                });
              }}
            >
              <SelectTrigger className={`w-1/3 min-w-[150px] ${isList ? 'flex-grow' : ''}`}>
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="string">string</SelectItem>
                <SelectItem value="float">float</SelectItem>
                <SelectItem value="int">int</SelectItem>
                <SelectItem value="boolean">boolean</SelectItem>
                <SelectItem value="list">list</SelectItem>
                <SelectItem value="dict">dict</SelectItem>
              </SelectContent>
            </Select>
            {!isList && (
              <Button variant="ghost" size="sm" onClick={() => removeItem(index)} className="p-1">
                <Trash2 size={16} />
              </Button>
            )}
            {item.type === 'list' && item.subType && (
              <div className="w-full mt-2 ml-4 flex items-center">
                <span className="mr-2 text-sm text-gray-500">List type:</span>
                <SchemaForm
                  schema={[item.subType as SchemaItem]}
                  onUpdate={(newSubSchema) => updateItem(index, { ...item, subType: newSubSchema[0] })}
                  level={0}
                  isList={true}
                />
              </div>
            )}
            {item.type === 'dict' && item.subType && (
              <div className="w-full mt-2 ml-4">
                <SchemaForm
                  schema={item.subType as SchemaItem[]}
                  onUpdate={(newSubSchema) => updateItem(index, { ...item, subType: newSubSchema })}
                  level={level + 1}
                />
              </div>
            )}
          </div>
        ))}
        {!isList && (
          <Button variant="outline" size="sm" onClick={addItem} className="mt-2">
            <Plus size={16} className="mr-2" /> Add Field
          </Button>
        )}
      </div>
    );
  };

// Settings Modal Component
const SettingsModal: React.FC<{
  opName: string;
  opType: string;
  isOpen: boolean;
  onClose: () => void;
  otherKwargs: Record<string, string>;
  onSettingsSave: (newSettings: Record<string, string>) => void;
}> = ({ opName, opType, isOpen, onClose, otherKwargs, onSettingsSave }) => {
  const [localSettings, setLocalSettings] = useState<Array<{ id: number; key: string; value: string }>>(
    Object.entries(otherKwargs).map(([key, value], index) => ({ id: index, key, value }))
  );

  useEffect(() => {
    setLocalSettings(Object.entries(otherKwargs).map(([key, value], index) => ({ id: index, key, value })));
  }, [otherKwargs]);

  if (!isOpen) return null;

  const handleSettingsChange = (id: number, newKey: string, newValue: string) => {
    setLocalSettings(prev => prev.map(setting => 
      setting.id === id ? { ...setting, key: newKey, value: newValue } : setting
    ));
  };

  const addSetting = () => {
    setLocalSettings(prev => [...prev, { id: prev.length, key: '', value: '' }]);
  };

  const removeSetting = (id: number) => {
    setLocalSettings(prev => prev.filter(setting => setting.id !== id));
  };

  const handleSave = () => {
    const newSettings = localSettings.reduce((acc, { key, value }) => {
      if (key !== '' && value !== '') {
        acc[key] = value;
      }
      return acc;
    }, {} as Record<string, string>);
    onSettingsSave(newSettings);
    onClose();
  };

  const isValidSettings = () => {
    const keys = localSettings.map(({ key }) => key);
    return localSettings.every(({ key, value }) => key !== '' && value !== '') &&
           new Set(keys).size === keys.length;
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>{opName}</DialogTitle>
          <DialogDescription>
            Add or modify additional arguments for this {opType} operation.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          {localSettings.map(({ id, key, value }) => (
            <div key={id} className="flex items-center gap-4">
              <Input
                className="flex-grow font-mono"
                value={key}
                onChange={(e) => handleSettingsChange(id, e.target.value, value)}
                placeholder="Key"
              />
              <Input
                className="flex-grow font-mono"
                value={value}
                onChange={(e) => handleSettingsChange(id, key, e.target.value)}
                placeholder="Value"
              />
              <Button variant="ghost" size="sm" onClick={() => removeSetting(id)}>
                <Trash2 size={15} />
              </Button>
            </div>
          ))}
          <Button onClick={addSetting}>Add Setting</Button>
        </div>
        <DialogFooter>
          <Button onClick={handleSave} disabled={!isValidSettings()}>Save</Button>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export const OperationCard: React.FC<{ 
  operation: Operation; 
  index: number; 
  onDelete: (id: string) => void;
  onUpdate: (id: string, updatedOperation: Operation) => void;
}> = ({ operation, index, onDelete, onUpdate }) => {
  const [schema, setSchema] = useState<SchemaItem[]>(
    operation.outputSchema 
      ? Object.entries(operation.outputSchema).map(([key, type]) => ({ key, type: type as SchemaType }))
      : []
  );
  const [isEditing, setIsEditing] = useState(false);
  const [editedName, setEditedName] = useState(operation.name);
  const [isSchemaExpanded, setIsSchemaExpanded] = useState(schema.length === 0);
  const [isGuardrailsExpanded, setIsGuardrailsExpanded] = useState(false);
  const [guardrails, setGuardrails] = useState<string[]>(operation.validate || []);
  const { setOutputs, setIsLoadingOutputs } = usePipelineContext();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [otherKwargs, setOtherKwargs] = useState<Record<string, string>>(operation.otherKwargs || {});

  const handleSchemaUpdate = (newSchema: SchemaItem[]) => {
    setSchema(newSchema);
    const newOutputSchema = newSchema.reduce((acc, item) => {
      acc[item.key] = item.type;
      return acc;
    }, {} as Record<string, string>);
    onUpdate(operation.id, { ...operation, outputSchema: newOutputSchema });
  };

  const handleNameEdit = () => {
    onUpdate(operation.id, { ...operation, name: editedName });
    setIsEditing(false);
  };

  const handleRunOperation = async () => {
    setIsLoadingOutputs(true);
    try {
      const response = await fetch('/debate_intermediates/extract_themes_and_viewpoints.json');
      let data = await response.json();
      if (data.length > 0 && 'date' in data[0]) {
        data.sort((a: { date: string }, b: { date: string }) => new Date(b.date).getTime() - new Date(a.date).getTime());
      }
      setOutputs(data);
    } catch (error) {
      console.error('Error fetching outputs:', error);
    } finally {
      setIsLoadingOutputs(false);
    }
  };

  const handleGuardrailChange = (index: number, value: string) => {
    const newGuardrails = [...guardrails];
    newGuardrails[index] = value;
    setGuardrails(newGuardrails);
    onUpdate(operation.id, { ...operation, validate: newGuardrails });
  };

  const addGuardrail = () => {
    const newGuardrails = [...guardrails, ''];
    setGuardrails(newGuardrails);
    onUpdate(operation.id, { ...operation, validate: newGuardrails });
  };

  const removeGuardrail = (index: number) => {
    const newGuardrails = guardrails.filter((_, i) => i !== index);
    setGuardrails(newGuardrails);
    onUpdate(operation.id, { ...operation, validate: newGuardrails });
  };

  const handleSettingsSave = (newSettings: Record<string, string>) => {
    setOtherKwargs(newSettings);
    onUpdate(operation.id, { ...operation, otherKwargs: newSettings });
  };

  return (
    <Draggable draggableId={operation.id} index={index}>
    {(provided) => (
      <Card className="mb-3 relative rounded-sm bg-white shadow-sm">
        <div
          className="absolute left-0 top-0 bottom-0 w-6 flex items-center justify-center cursor-move hover:bg-gray-100"
          {...provided.dragHandleProps}
        >
          <GripVertical size={16} />
        </div>
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          className="ml-6"
        >
          <CardHeader className="flex justify-between items-center py-3 px-4">
            <div className="flex space-x-2 absolute left-8 top-3">
              <Button variant="ghost" size="sm" className="p-1 h-7 w-7" onClick={() => setIsSettingsOpen(true)}>
                <Settings size={15} className="text-gray-500" />
              </Button>
              <Button variant="ghost" size="sm" className="p-1 h-7 w-7">
                <Zap size={15} className="text-yellow-500" />
              </Button>
              <Button variant="ghost" size="sm" className="p-1 h-7 w-7" onClick={handleRunOperation}>
                <Play size={15} className="text-green-500" />
              </Button>
            </div>
            {isEditing ? (
              <Input
                value={editedName}
                onChange={(e) => setEditedName(e.target.value)}
                onBlur={handleNameEdit}
                onKeyPress={(e) => e.key === 'Enter' && handleNameEdit()}
                className="text-sm font-medium w-1/2 font-mono"
                autoFocus
              />
            ) : (
              <span 
                className={`text-sm font-medium cursor-pointer ${operation.llmType === 'LLM' ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-transparent bg-clip-text' : ''}`}
                onClick={() => setIsEditing(true)}
              >
                {operation.name} ({operation.type})
              </span>
            )}
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => onDelete(operation.id)}
              className="hover:bg-red-100 absolute top-3 right-3 p-1 h-7 w-7"
            >
              <Trash2 size={15} className="text-red-500" />
            </Button>
          </CardHeader>
          <CardContent className="py-3 px-4">
          {operation.llmType === 'LLM' && (
              <>
                <Textarea placeholder="Enter prompt" className="mb-3 rounded-sm text-sm font-mono" rows={3} />
                <div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsSchemaExpanded(!isSchemaExpanded)}
                    className="mb-2 p-0"
                  >
                    <ChevronDown
                      size={16}
                      className={`mr-1 transition-transform duration-200 ${
                        isSchemaExpanded ? 'transform rotate-180' : ''
                      }`}
                    />
                    <h4 className="text-sm font-semibold">Output Types</h4>
                  </Button>
                  {isSchemaExpanded && (
                    <SchemaForm schema={schema} onUpdate={handleSchemaUpdate} />
                  )}
                </div>
              </>
            )}
          </CardContent>
          <div className="border-t border-red-500">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsGuardrailsExpanded(!isGuardrailsExpanded)}
              className="w-full text-red-500 hover:bg-red-50 flex justify-between items-center"
            >
              <span>Guardrails ({guardrails.length})</span>
              <ChevronDown
                size={16}
                className={`transition-transform duration-200 ${
                  isGuardrailsExpanded ? 'transform rotate-180' : ''
                }`}
              />
            </Button>
            {isGuardrailsExpanded && (
              <div className="p-4 bg-red-50">
                {guardrails.map((guardrail, index) => (
                  <div key={index} className="flex items-center mb-2">
                    <Input
                      value={guardrail}
                      onChange={(e) => handleGuardrailChange(index, e.target.value)}
                      placeholder="Enter guardrail"
                      className="flex-grow text-sm text-red-700 font-mono"
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeGuardrail(index)}
                      className="ml-2 p-1 h-7 w-7 hover:bg-red-100"
                    >
                      <Trash2 size={15} className="text-red-500" />
                    </Button>
                  </div>
                ))}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={addGuardrail}
                  className="mt-2 text-red-500 border-red-500 hover:bg-red-100"
                >
                  <Plus size={16} className="mr-2" /> Add Guardrail
                </Button>
              </div>
            )}
          </div>

          <SettingsModal
            opName={operation.name}
            opType={operation.type}
            isOpen={isSettingsOpen}
            onClose={() => setIsSettingsOpen(false)}
            otherKwargs={otherKwargs}
            onSettingsSave={handleSettingsSave}
          />

        </div>
      </Card>
    )}
  </Draggable>
  );
};

