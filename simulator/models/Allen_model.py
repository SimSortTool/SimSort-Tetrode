
import os
import json
import neuron
import LFPy

class AllenModel:
    def __init__(
            self, 
            cell_model_folder, 
            dt = 2**-5, 
            start_T = 0, 
            end_T = 1, # s
            **kwargs):

        for key, value in kwargs.items():  
            setattr(self, key, value)  
        
        self.cell_model_folder = os.path.abspath(cell_model_folder)
        self.dt = dt
        self.start_T = start_T
        self.end_T = end_T
        self.cwd = os.getcwd()
        self.params = None

    def compile_mechanisms(self):
        os.chdir(self.cell_model_folder)
        mod_folder = "modfiles"
        if mod_folder not in os.listdir():
            os.mkdir(mod_folder)
            
        os.chdir(mod_folder)
        os.system('nrnivmodl')
        os.chdir('..')
        neuron.load_mechanisms(mod_folder)

    def load_parameters(self):
        with open(os.path.join(self.cell_model_folder, "fit_parameters.json"), 'r') as f:
            self.params = json.load(f)

    def create_LFPy_cell(self):
        celsius = self.params["conditions"][0]["celsius"]
        v_init = self.params["conditions"][0]["v_init"]
        active_mechs = self.params["genome"]
        reversal_potentials = self.params["conditions"][0]["erev"]
        neuron.h.celsius = celsius

        cell_parameters = {
            'morphology': 'reconstruction.swc',
            'v_init': v_init,
            'passive': False,
            # 'passive_parameters' : {'g_pas' : 0.001, 'e_pas' : -65.},
            'nsegs_method': 'lambda_f',
            'lambda_f': 1., #hz default 100
            'dt': self.dt,
            'pt3d': True,
            'tstart': self.start_T,
            'tstop': self.end_T,
            'delete_sections' : True,
            'verbose' : False
        }
        
        cell = LFPy.Cell(**cell_parameters)

        for sec in neuron.h.allsec():
            sec.insert("pas")
            sectype = sec.name().split("[")[0]
            for sec_dict in active_mechs:
                if sec_dict["section"] == sectype:
                    # print(sectype, sec_dict)
                    if not sec_dict["mechanism"] == "":
                        sec.insert(sec_dict["mechanism"])
                    exec ("sec.{} = {}".format(sec_dict["name"], sec_dict["value"]))

            for sec_dict in reversal_potentials:
                if sec_dict["section"] == sectype:
                    # print(sectype, sec_dict)
                    for key in sec_dict.keys():
                        if not key == "section":
                            exec ("sec.{} = {}".format(key, sec_dict[key]))
   
        return cell
    
    def run(self):
        neuron.h("forall delete_section()")
        self.compile_mechanisms()
        self.load_parameters()
        return self.create_LFPy_cell()


    def __enter__(self):
        os.chdir(self.cell_model_folder)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.cwd)


if __name__ == "__main__":
    cell_model_folder = os.path.join('allen_models', 'neuronal_model_488462965')
    neuron_model = AllenModel(cell_model_folder)
    with neuron_model:
        cell = neuron_model.run()
