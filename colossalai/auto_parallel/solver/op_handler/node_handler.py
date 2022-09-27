from abc import ABC, abstractmethod
from torch.fx.node import Node
from colossalai.device.device_mesh import DeviceMesh
from colossalai.tensor.shape_consistency import ShapeConsistencyManager
from typing import Dict, List
from ..sharding_strategy import ShardingStrategy_V2, StrategiesVector, OperationData, TrainCycleItem
from ..strategy import StrategyGenerator_V2


class NodeHandler(ABC):
    '''
    The NodeHandler is an abstract class used to generate every possible strategies for an operator node.

    Args:
        node (Node): the input node in node argument list.
        device_mesh (DeviceMesh): A logical view of a physical mesh.
        strategies_vector (StrategiesVector): all the strategies generated in this handler will be recorded into the strategies_vector.
    '''

    def __init__(
        self,
        node: Node,
        device_mesh: DeviceMesh,
        strategies_vector: StrategiesVector,
    ) -> None:
        self.node = node
        self.predecessor_node = list(node._input_nodes.keys())
        self.successor_node = list(node.users.keys())
        self.device_mesh = device_mesh
        self.strategies_vector = strategies_vector

    def update_resharding_cost(self, strategy: ShardingStrategy_V2) -> None:
        """
        Compute the resharding costs and save the costs in the ShardingStrategy object.
        """
        # TODO: test this function when other handlers are ready
        resharding_costs = {}
        shape_consistency_manager = ShapeConsistencyManager()
        for node in self.predecessor_node:
            node_name = str(node)

            # get the sharding specs for this node generated
            # in its own node handler
            assert hasattr(node, 'strategies_vector'), \
                f'The predecessor node {node_name} has no strategy vector to compute the resharding cost.'
            prev_strategy_vector = node.strategies_vector
            prev_sharding_specs = [strategy.get_sharding_spec_by_name(node_name) for strategy in prev_strategy_vector]

            # get the current sharding spec generated by this node handler
            op_data = strategy.get_op_data_by_name(node_name)
            current_sharding_spec = strategy.sharding_specs[op_data]

            # create data structrure to store costs
            if op_data not in resharding_costs:
                resharding_costs[op_data] = {}

            # for each sharding spec generated by the predecessor's node handler
            # compute the resharding cost to switch to the sharding spec generated
            # by the current node handler
            for prev_sharding_spec in prev_sharding_specs:
                fwd_cost = shape_consistency_manager.shape_consistency(prev_sharding_spec, current_sharding_spec)
                bwd_cost = shape_consistency_manager.shape_consistency(current_sharding_spec, prev_sharding_spec)
                resharding_cost = TrainCycleItem(fwd=fwd_cost, bwd=bwd_cost, total=fwd_cost + bwd_cost)
                resharding_costs[op_data][prev_sharding_spec] = resharding_cost
        strategy.resharding_costs = resharding_costs

    def register_strategy(self, compute_resharding_cost: bool = False) -> StrategiesVector:
        """
        Register different sharding strategies for the current node.
        """
        strategy_generators = self.get_strategy_generator()
        for generator in strategy_generators:
            strategies = generator.generate()

            # compute the resharding costs based on the previous node
            # strategies if specified
            if compute_resharding_cost:
                strategies = list(map(self.update_resharding_cost, strategies))
            self.strategies_vector.extend(strategies)

        strategies_vector = map(self.post_process, self.strategies_vector)
        self.strategies_vector = list(strategies_vector)
        return self.strategies_vector

    def post_process(self, strategy: ShardingStrategy_V2):
        # tranform the strategy generated
        # e.g. to process the sharding strategy for the transposed weights
        return strategy

    @abstractmethod
    def get_strategy_generator(self) -> List[StrategyGenerator_V2]:
        """
        Define which generators should be used by this NodeHandler object.
        """
        pass

    @abstractmethod
    def get_operation_data_mapping(self) -> Dict[str, OperationData]:
        """
        Returns the mapping between the logical operation data to its physical data.
        A logical operation data is a data associated with an operation, which can be input and output. It is 
        defined by the strategy generator, for example, a matrix multiplication operation has two operands "input" 
        and "other" and one result "output". For a nn.Linear module, the physical operand for "input" is
        the module input, the physical operand for "other" is the module weight, and the physical result for "output"
        is the module output.
        Note that the operand name is specified by the StrategyGenerator object.

        For example:

            # for a linear layer
            mapping = {
                "input": Operand(name=str(self.node.args[0]), type=OperationDataType.ARG, data=self.node.args[0]._meta_data),
                "other": Operand(name="weight", type=OperationDataType.PARAM, data=self.named_parameters['weight']),
                "bias": Operand(name="bias", type=OperationDataType.PARAM, data=self.named_parameters['bias']),
                "output": Operand(name=str(self.node), type=OperationDataType.OUTPUT, data=self.node._meta_data),
            }
        """
        pass


class ModuleHandler(NodeHandler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        print("created")

        # set attributes to access module parameters for convenience
        assert self.node.graph.owning_module is not None, \
            f'The graph is not associated with a module, please make sure it can be used to instantiate a GraphModule object.'
        module = self.node.graph.owning_module.get_submodule(self.node.target)
        named_parameters = list(module.named_parameters(recurse=False))
        # convert named parameters from list to dict
        named_parameters = {k: v for k, v in named_parameters}
        self.module = module
        self.named_parameters = named_parameters
