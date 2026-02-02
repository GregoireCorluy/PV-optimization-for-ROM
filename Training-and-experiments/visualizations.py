from utils import *
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import plotly.io as pio

def visualize_maxWiYi(path_data, path_state, model, idx_species_removed, save = False, path_name_save = None, abs = True, file_species_names = "Xu-state-space-names.csv", header = None):
    """
    Visualize max(|w_i|*Y_i) for every species to see which species have the most impact on the PV.
    """
    
    final_weights = model.get_total_encoder_weights(npy = True)[0]

    #get index spcies removed
    my_species = Species(path_data, file_species_names)
    
    list_species = my_species.get_list_species()
    for index in sorted(idx_species_removed, reverse=True):
        list_species.pop(index)

    #get mass fractions of species
    state_space = read_csv(path_state, header = header)
    state_space = state_space.drop(state_space.columns[idx_species_removed], axis=1)
    state_space_matrix = state_space.to_numpy()

    wi_Yi = state_space_matrix*np.abs(final_weights) #|w_i|*Y_i
    if(abs):
        max_wi_Yi = np.max(wi_Yi, axis = 0) #take max of every species
    else:
        max_wi_Yi = np.sign(final_weights)*np.max(wi_Yi, axis = 0) #take max of every species
    max_wi_Yi_scaled = max_wi_Yi/np.max(np.abs(max_wi_Yi))

    latex_species_list = [s.replace('2', r'$_2$') for s in list_species]
    latex_species_list = [s.replace('3', r'$_3$') for s in latex_species_list]

    plt.scatter(np.arange(len(max_wi_Yi_scaled)), max_wi_Yi_scaled, color = "k", s=10)
    if(abs):
        plt.ylabel(r"max($|w_i| \cdot Y_i$) [$-$]",fontsize = 15)
    else:
        plt.ylabel(r"max($w_i \cdot Y_i$) [$-$]",fontsize = 15)
    plt.xticks(np.arange(len(max_wi_Yi_scaled)), latex_species_list, rotation = 90)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    if(save):
        plt.savefig(path_name_save, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_Wi(path_data, model, idx_species_removed, save = False, path_name_save = None, file_species_names = "Xu-state-space-names.csv"):
    """
    Visualize w_i for every species to see which species have the most impact on the PV.
    """

    """
    final_weights = model.encoder_species.weight.detach().numpy()[0]

    if(model.auto_scaling):
        final_weights *= model.scaling_weights.detach().numpy()
    """
    
    final_weights = model.get_total_encoder_weights(npy = True)[0]

    #get index spcies removed
    my_species = Species(path_data, file_species_names)
    
    list_species = my_species.get_list_species()
    for index in sorted(idx_species_removed, reverse=True):
        list_species.pop(index)

    latex_species_list = [s.replace('2', r'$_2$') for s in list_species]
    latex_species_list = [s.replace('3', r'$_3$') for s in latex_species_list]

    plt.scatter(np.arange(len(final_weights)), final_weights, color = "k", s=10)
    plt.ylabel(r"$w_i$ [$-$]",fontsize = 15)
    plt.xticks(np.arange(len(final_weights)), latex_species_list, rotation = 90)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    if(save):
        plt.savefig(path_name_save, dpi=300, bbox_inches='tight')
    plt.show()

###########
#For 2 PV's
###########

def visualize_maxWiYi_2PV(path_data, path_state, model, idx_species_removed, save = False, path_name_save = None, dpi = 300, abs = True):
    """
    Visualize max(|w_i|*Y_i) for every species to see which species have the most impact on the PV.
    """

    final_weights_1 = model.get_total_encoder_weights(npy = True)[0]
    final_weights_2 = model.get_total_encoder_weights(npy = True)[1]

    #get index spcies removed
    my_species = Species(path_data)
    
    list_species = my_species.get_list_species()
    for index in sorted(idx_species_removed, reverse=True):
        list_species.pop(index)

    #get mass fractions of species
    state_space = read_csv(path_state, header = None)
    state_space = state_space.drop(state_space.columns[idx_species_removed], axis=1)
    state_space_matrix = state_space.to_numpy()

    wi_Yi_1 = state_space_matrix*np.abs(final_weights_1) #|w_i|*Y_i
    if(abs):
        max_wi_Yi_1 = np.max(wi_Yi_1, axis = 0) #take max of every species
    else:
        max_wi_Yi_1 = np.sign(final_weights_1)*np.max(wi_Yi_1, axis = 0) #take max of every species
    max_wi_Yi_scaled_1 = max_wi_Yi_1/np.max(np.abs(max_wi_Yi_1))

    wi_Yi_2 = state_space_matrix*np.abs(final_weights_2) #|w_i|*Y_i
    if(abs):
        max_wi_Yi_2 = np.max(wi_Yi_2, axis = 0) #take max of every species
    else:
        max_wi_Yi_2 = np.sign(final_weights_2)*np.max(wi_Yi_2, axis = 0) #take max of every species
    max_wi_Yi_scaled_2 = max_wi_Yi_2/np.max(np.abs(max_wi_Yi_2))

    latex_species_list = [s.replace('2', r'$_2$') for s in list_species]
    latex_species_list = [s.replace('3', r'$_3$') for s in latex_species_list]

    plt.scatter(np.arange(len(max_wi_Yi_scaled_1)), max_wi_Yi_scaled_1, color = "r", s=15, label = "PV1", marker='o')
    plt.scatter(np.arange(len(max_wi_Yi_scaled_2)), max_wi_Yi_scaled_2, color = "b", s=15, label = "PV2", marker='^')
    if(abs):
        plt.ylabel(r"max($|w_i| \cdot Y_i$) [$-$]",fontsize = 15)
    else:
        plt.ylabel(r"max($w_i \cdot Y_i$) [$-$]",fontsize = 15)
    plt.xticks(np.arange(len(max_wi_Yi_scaled_1)), latex_species_list, rotation = 90)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend()
    
    if(save):
        plt.savefig(path_name_save, dpi=dpi)
    plt.show()

def visualize_Wi_2PV(path_data, model, idx_species_removed, save = False, path_name_save = None, dpi = 300):
    """
    Visualize w_i for every species to see which species have the most impact on the PV.
    """

    final_weights_1 = model.encoder_species.weight.detach().numpy()[0]
    final_weights_2 = model.encoder_species.weight.detach().numpy()[1]

    #get index spcies removed
    my_species = Species(path_data)
    
    list_species = my_species.get_list_species()
    for index in sorted(idx_species_removed, reverse=True):
        list_species.pop(index)

    latex_species_list = [s.replace('2', r'$_2$') for s in list_species]
    latex_species_list = [s.replace('3', r'$_3$') for s in latex_species_list]

    plt.scatter(np.arange(len(final_weights_1)), final_weights_1, color = "r", s=15, label = "PV1", marker = 'o')
    plt.scatter(np.arange(len(final_weights_2)), final_weights_2, color = "b", s=15, label = "PV2", marker = '^')
    plt.ylabel(r"$w_i$ [$-$]",fontsize = 15)
    plt.xticks(np.arange(len(final_weights_1)), latex_species_list, rotation = 90)
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    plt.legend()

    if(save):
        plt.savefig(path_name_save, dpi=dpi)
    plt.show()

def showManifold(mf, PV, colorData, colorName, pathSave = None, save = False, dpi = 300, idxPV = 0, idxPVsource = 0, clean = False):

    PVmultiple = PV.shape[1]>1

    PVshown = PV[:, idxPV] if PVmultiple else PV
    PVsourceShown = colorData[:, idxPVsource] if PVmultiple else colorData
    PVnumber = str(idxPV+1) if PVmultiple else ""

    plt.scatter(mf, PVshown, c=PVsourceShown, s=1, cmap='viridis', marker='o')  # You can adjust the color and marker style
    if(not clean):
        plt.title('PV vs mixture fraction')
        cbar = plt.colorbar()
        cbar.set_label(f'{colorName}', fontsize=15) 
        plt.xlabel('f', fontsize = 15)
        plt.ylabel(f'PV{PVnumber}', fontsize = 15)
        plt.grid(False)
    else:
        plt.xticks([])
        plt.yticks([])
        if(save):
            pathSave = pathSave.replace('.png', '_clean.png')
    if(save):
        plt.savefig(pathSave, dpi=dpi)
    plt.show()

def showPVPV(PV, PVsource, pathSave, save = False, dpi = 300, idxPVsource = 0, clean = False):

    PVmultiple = PV.shape[1]>1

    PVsourceShown = PVsource[:, idxPVsource] if PVmultiple else PVsource
    PVsourceNumber = str(idxPVsource+1) if PVmultiple else ""

    plt.scatter(PV[:,0], PV[:,1], c=PVsourceShown, s=1, cmap='viridis', marker='o')  # You can adjust the color and marker style
    if(not clean):    
        plt.title('PV1 vs PV2')
        cbar = plt.colorbar()
        cbar.set_label(f'PV source {PVsourceNumber}', fontsize=15)  
        plt.xlabel('PV1', fontsize = 15)
        plt.ylabel('PV2', fontsize = 15)
        plt.grid(False)
    else:
        pathSave = pathSave.replace('.png', '_clean.png')
    if(save):
        plt.savefig(pathSave, dpi=dpi)
    plt.show()

def showCurve(training, validation, pathSave, save = False, dpi = 300, color = True):

    color1 = "blue" if color else "k"
    color2 = "red" if color else "k"

    if(color):
        pathSave = pathSave.replace('.png', '_color.png')

    plt.plot(training, label='Training curve', color=color1)
    plt.plot(validation, label='Validation curve', linestyle='-.', color=color2)
    plt.xlabel('Epoch', fontsize = 15)
    plt.ylabel('Loss', fontsize = 15)
    plt.yscale('log')
    plt.legend()
    plt.title('Training and validation curves')
    plt.grid(True)
    if(save):
        plt.savefig(pathSave, dpi=dpi)
    plt.show()

def showManifold3D(mf, PV, PV_source, idxPVsource):

    pio.renderers.default = "notebook"

    print("Plot figure")
    fig = go.Figure(data=[go.Scatter3d(
            x=mf,
            y=PV[:,0],
            z=PV[:,1],
            mode='markers',
            marker=dict(
                size=2,
                color=PV_source[:,idxPVsource],
                colorscale='inferno',
                opacity=0.8
            )
        )])
        
    fig.update_layout(autosize=True,
                    width=600, height=600,
                    scene = dict(
                    xaxis_title='f',
                    yaxis_title='PV1',
                    zaxis_title='PV2'))

    fig.show()