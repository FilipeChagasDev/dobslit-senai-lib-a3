# Filipe Chagas, 2023

# Importing standard Qiskit libraries
import qiskit
from qiskit import Aer
import pandas as pd
from typing import *

def organize_qiskit_result(result_counts: Dict[str, int], registers_names: List[str]) -> pd.DataFrame:
    """Organize the results of the execution of a quantum circuit in a DataFrame.

    :param result_counts: Counts dict returned by Qiskit.
    :type result_counts: Dict[str, int]
    :param registers_names: List with the names of the registers.
    :type registers_names: List[str]
    :param registers_dtypes: List with the data types of the registers. 
    :type registers_dtypes: List[QQDTypes]
    :return: DataFrame containing the value of each register in each outcome and the frequencies of the outcomes.
    :rtype: pd.DataFrame
    """
    out_dict = {key:[] for key in registers_names+['$freq']}
    
    for full_bit_string in result_counts.keys():
        freq = result_counts[full_bit_string] #absolute frequency of the current bit-string
        reg_bit_string = full_bit_string.split(' ') #separate registers
        
        #Convert bit-strings to naturals, integers or booleans
        reg_data_list = []
        for i in range(len(reg_bit_string)):
            reg_data_list.append(reg_bit_string[i])
            
        #Append results to the output dictionary
        for i in range(len(reg_bit_string)):
            out_dict[registers_names[i]].append(reg_data_list[-1-i])

        out_dict['$freq'].append(freq)
        
    return pd.DataFrame(out_dict, ).sort_values('$freq', ascending=False).reset_index(drop=True)


class GroverAlgorithm():
    def __init__(self):
        self.circ = qiskit.QuantumCircuit()
        
        #Cria um qubit auxiliar do algoritmo de Grover
        self.phase_ancilla = qiskit.QuantumRegister(1, name='phase_ancilla')
        self.circ.add_register(self.phase_ancilla)
        self.circ.h(self.phase_ancilla[0])
        
        self.qubits = [] #Lista que deve conter os qubits usados (exceto o phase_ancilla)
        self.clbits = [] #Lista que deve conter os bits clássicos
        self.names = [] #Lista de nomes dos qubits
        self.prepare()
        
    def create_qubit(self, label: str):
        #Este método cria e retorna um registrador de 1 qubit
        one_qubit_reg = qiskit.QuantumRegister(1, name=f'q_{label}')
        one_bit_reg = qiskit.ClassicalRegister(1, name=f'c_{label}')
        self.circ.add_register(one_qubit_reg)
        self.circ.add_register(one_bit_reg)
        self.qubits.append(one_qubit_reg)
        self.clbits.append(one_bit_reg)
        self.names.append(label)
        return one_qubit_reg
    
    def prepare(self):
        raise NotImplementedError()
        
    def h(self, target_qubit):
        #Adiciona uma porta Hadamard ao circuito
        self.circ.h(target_qubit[0])
        
    def z(self, target_qubit):
        #Adiciona uma porta Pauli-Z ao circuito
        self.circ.z(target_qubit[0])
        
    def cz(self, ctrl_qubit, target_qubit):
        #Adiciona uma porta Controlled-Z ao circuito
        self.circ.cz(ctrl_qubit[0], target_qubit[0])
        
    def mcx(self, ctrl_qubits, target_qubit):
        #Adiciona uma porta Multi-controlled-X ao circuito
        self.circ.mcx([qbit[0] for qbit in ctrl_qubits], target_qubit[0])
    
    def x(self, target_qubit):
        #Adiciona uma porta Pauli-X ao circuito
        self.circ.x(target_qubit[0])
        
    def logic_not(self, target_qubit):
        #Adiciona um equivalente quântico da porta NOT ao circuito
        self.circ.x(target_qubit[0])
    
    def logic_and(self, operand_qubits, target_qubit):
        #Adiciona um equivalente quântico da porta AND ao circuito
        self.circ.mcx([qbit[0] for qbit in operand_qubits], target_qubit[0])
        
    def logic_or(self, operand_qubits, target_qubit):
        #Adiciona um equivalente quântico da porta OR ao circuito
        operand_qubits = [qbit[0] for qbit in operand_qubits]
        self.circ.x(operand_qubits)
        self.circ.mcx(operand_qubits, target_qubit[0])
        self.circ.x(operand_qubits)
        self.circ.x(target_qubit[0])
    
    def build_search_space(self):
        #Método abstrato onde o circuito que constroi o espaço de busca deve ser construido
        raise NotImplementedError()
    
    def revert_search_space(self):
        #Método abstrato onde o circuito inverso do que constroi o espaço de busca deve ser construido
        raise NotImplementedError()
        
    def build_all(self, target_qubit, n_iterations: int):
        #Método que constroi o circuito completo da busca de Grover
        for i in range(n_iterations):
            #--- constroi o espaço de busca e marca as soluções ---
            self.circ.barrier()
            self.build_search_space()
            self.circ.barrier()
            self.cz(target_qubit, self.phase_ancilla)
            self.circ.barrier()
            self.revert_search_space()
            self.circ.barrier()
            
            #--- constroi o circuito do operador 2|0><0|-I ---
            non_target_qubits = [qbit[0] for qbit in self.qubits if qbit != target_qubit] #lista com todos os qubits exceto o qubit alvo
            for qbit in self.qubits:
                self.x(qbit)
            self.z(self.phase_ancilla)
            self.mcx(self.qubits, self.phase_ancilla)
            self.z(self.phase_ancilla)
            for qbit in self.qubits:
                self.x(qbit)
        self.circ.barrier()
        self.build_search_space()
        
        #--- adiciona portas de medição ---
        self.circ.barrier()
        for i in range(len(self.names)):
            self.circ.measure(self.qubits[i], self.clbits[i])
            
    def simulate(self):
        #Faz uma simulação e retorna os resultados como um dataframe
        sim = Aer.get_backend('aer_simulator_statevector')
        job = sim.run(self.circ, shots=1024)
        res = job.result()
        counts = res.get_counts()
        return organize_qiskit_result(counts, registers_names=self.names)