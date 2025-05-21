import os
import json
import neuron
import LFPy
import glob
from pathlib import Path
from tools import compile_all_mechanisms

class BBPModel:
    def __init__(
            self, 
            cell_model_folder, 
            dt=2**-6, 
            start_T=-159,
            end_T=10,
            v_init=-70,
            celsius=34,
            compile_mech=True,
            script_dir=os.getcwd(),
            **kwargs
            ):
        self.cell_model_folder = os.path.abspath(cell_model_folder)
        self.template_name = str(self.get_templatename())
        self.dt = dt
        self.start_T = start_T
        self.end_T = end_T
        self.cwd = script_dir
        self.params = None
        self.cell = None
        self.v_init = v_init
        self.celsius = celsius
        self.compile_mech = compile_mech

    def load_mechanisms(self):
        neuron.h.load_file("stdrun.hoc")
        neuron.h.load_file("import3d.hoc")
        neuron.load_mechanisms(str(Path(self.cell_model_folder).parent / "mods"))
        os.chdir(self.cell_model_folder)
        with open("biophysics.hoc", 'r') as f:
            biophysics = self.get_templatename()
        f.close()

        with open("morphology.hoc", 'r') as f:
            morphology = self.get_templatename()
        f.close()

        # get synapses template name
        synapses_file = str(Path("synapses") / "synapses.hoc")
        with open(synapses_file, 'r') as f:
            synapses = self.get_templatename()
        f.close()

        neuron.h.load_file("constants.hoc")
        if not hasattr(neuron.h, morphology):
            neuron.h.load_file(1, "morphology.hoc")

        if not hasattr(neuron.h, biophysics.split('_')[0] + '_biophys'):
            neuron.h.load_file(1, "biophysics.hoc")

        if not hasattr(neuron.h, synapses):
            # load synapses
            neuron.h.load_file(1, synapses_file)

        if not hasattr(neuron.h, self.template_name):
            neuron.h.load_file(1, "template.hoc")

    def get_templatename(self):
        os.chdir(self.cell_model_folder)
        with open("template.hoc", 'r') as f:
            for line in f.readlines():
                if "begintemplate" in line.split():
                    return line.split()[-1]

    def create_LFPy_cell(self):
        os.chdir(self.cell_model_folder)
        cell_parameters = {
            'morphology' : glob.glob('./morphology/*.asc')[0],
            'templatefile'  : str(Path("template.hoc").absolute()),
            'templatename' : self.template_name,
            'templateargs' : 0,
            'passive' : False,
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 1, #hz default 100
            'dt' : self.dt,  # timestamps of simulation in ms
            'v_init' : self.v_init,
            'celsius': self.celsius,
            'pt3d' : True,
            'tstart': self.start_T,  # start time of simulation, ms
            'tstop': self.end_T,
            'delete_sections' : True,
            'verbose' : False
        }
        # LFPy.cell.neuron.h("forall delete_section()")
        self.cell = LFPy.TemplateCell(**cell_parameters)
        return self.cell

    def run(self):
        os.chdir(self.cell_model_folder)
        cell_model_total = Path(self.cell_model_folder).parent
        compile_all_mechanisms(cell_model_total)
        self.load_mechanisms()
        print(self.template_name)
    
        return self.create_LFPy_cell()

    def __enter__(self):
        os.chdir(self.cell_model_folder)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)

if __name__ == "__main__":
    script_dir = os.getcwd()
    cell_model_folder = os.path.join('BBP_models', 'L5_BP_bAC217_1')
    neuron_model = BBPModel(cell_model_folder)
    with neuron_model:
        cell = neuron_model.run()
        print(cell)

