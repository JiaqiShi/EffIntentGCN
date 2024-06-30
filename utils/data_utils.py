import numpy as np
from scipy.spatial.transform import Rotation as R

JAAD_CONNECTS = [(0, 0), (0, 1), (0,14), (0,15), (1,2), (1,5), (2,3), (2,8), (3,4), (5,6), (5,11), (6,7), (8,9), (9,10),
                         (11, 12), (12, 13), (14,16), (15,17)]

def skeleton2dirvec(skeleton, connects, normalization=False):
    if len(connects[0]) == 3:
        pairs = [c[0:2] for c in connects]
    else:
        pairs = connects
    target_index = [i for (i,j) in pairs]
    source_index = [j for (i,j) in pairs]
    dirvec = skeleton[:, target_index, :] - skeleton[:, source_index, :]
    if normalization:
        l2 = np.linalg.norm(dirvec, ord=2, axis=2, keepdims=True)
        dirvec = dirvec/(l2 + 1e-12)
    return dirvec

def dirvec2euler(dirvec, initial_direction=0):
    '''
    Transform direction vector (bone vector) to rotation angle. right hand system and rotation order: x -> y -> z
    initial_direction: initial joint angle. if 0 upper vertical, 1 lower vertical, 2 right horizontal, 3 left horizontal.
    '''
    # rotation order: x -> y -> z
    x, y, z = dirvec 

    delta = 1e-6

    length_vector = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)
    if initial_direction == 0:
        rotation_x = np.rad2deg(np.arctan(-y/(z+delta)))
        rotation_y = np.rad2deg(np.arcsin(x/(length_vector+delta)))
        rotation_z = np.zeros_like(rotation_x)
    elif initial_direction == 1:
        rotation_x = np.rad2deg(np.arctan(y/-(z+delta)))
        rotation_y = np.rad2deg(np.arcsin(-x/(length_vector+delta)))
        rotation_z = np.zeros_like(rotation_x)
    elif initial_direction == 2:
        rotation_x = np.rad2deg(np.arctan(z/(y+delta)))
        rotation_y = np.zeros_like(rotation_x)
        rotation_z = np.rad2deg(np.arcsin(x/(length_vector+delta)))
    elif initial_direction == 3:
        rotation_x = np.rad2deg(np.arctan(z/(y+delta)))
        rotation_y = np.zeros_like(rotation_x)
        rotation_z = -np.rad2deg(np.arcsin(x/(length_vector+delta)))

    return [rotation_x, rotation_y, rotation_z]

def representation_trans(s_data, source_type, target_type, dataset=None):
    '''
    Transform gesture motion representation from source type to target type.
    source_type: str. skeleton/dirvec/dirvec_norm/euler
    target_type: str. dirvec/dirvec_norm/euler/quat/rotvec/matrix/relative/temp_disp/relative_disp/relative_disp_all
    dataset: str. JAAD
    '''
    assert s_data.shape[-1] in (2, 3) and len(s_data.shape) == 3
    T, N_joint, dim = s_data.shape

    connects = JAAD_CONNECTS

    if target_type in ['relative', 'temp_disp', 'relative_disp', 'relative_disp_all']:
        assert source_type == 'skeleton'
        if target_type == 'relative':
            return skeleton2relative_coordinates(s_data, references=(1, 4, 11, 14))
        elif target_type == 'temp_disp':
            return skeleton2displacement(s_data)
        elif target_type == 'relative_disp': 
            return np.concatenate([skeleton2relative_coordinates(s_data, references=(1, 4, 11, 14)), skeleton2displacement(s_data)], 2)
        elif target_type == 'relative_disp_all':
            return np.concatenate([s_data, skeleton2relative_coordinates(s_data, references=(1, 4, 11, 14)), skeleton2displacement(s_data)], 2)

    if source_type == 'skeleton':
        if target_type == 'dirvec':
            return skeleton2dirvec(s_data, connects)
        elif target_type == 'dirvec_norm':
            return skeleton2dirvec(s_data, connects, normalization=True)
        # to direction vector
        s_data = skeleton2dirvec(s_data, connects, normalization=True)
    
    if source_type in ['skeleton','dirvec','dirvec_norm']:
        s_data = np.transpose(s_data, (1,2,0))
        s_data = np.transpose(np.array([dirvec2euler(d_ontjoint, connect[-1]) for d_ontjoint, connect in zip(s_data, connects)]), (2,0,1))
        if target_type == 'euler':
            return s_data

    s_data = R.from_euler('xyz', s_data.reshape(T*N_joint,-1), degrees=True)

    if target_type == 'quat':
        return s_data.as_quat().reshape(T, N_joint, -1)
    elif target_type == 'rotvec':
        return s_data.as_rotvec().reshape(T, N_joint, -1)
    elif target_type == 'matrix':
        return s_data.as_matrix().reshape(T, N_joint, -1)
    else:
        raise ValueError(f'Transform to {target_type} not existing.')
    
def skeleton2relative_coordinates(skeleton,
                             references=(1, 4, 11, 14)):
    # time step, joint number, feature number
    T, V, C = skeleton.shape

    out_repre = np.zeros((T, V, len(references)*C))

    valid_frame = (skeleton != 0).sum((1,2)) > 0
    start_frame = valid_frame.argmax()
    end_frame = T - valid_frame[::-1].argmax()

    valid_data = skeleton[start_frame:end_frame, :, :]

    rel_coords = [valid_data - np.expand_dims(valid_data[:, i, :], 1) for i in references]

    rel_coords = np.concatenate(rel_coords, 2)

    out_repre[start_frame:end_frame, :, :] = rel_coords

    return out_repre

def skeleton2displacement(skeleton):
    # time step, joint number, feature number
    T, V, C = skeleton.shape

    out_repre = np.zeros((T, V, C))

    valid_frame = (skeleton != 0).sum((1,2)) > 0
    start_frame = valid_frame.argmax()
    end_frame = T - valid_frame[::-1].argmax()

    valid_data = skeleton[start_frame:end_frame, :, :]

    temporal_disp = valid_data[1:] - valid_data[:-1]
    out_repre[start_frame:end_frame-1, :, :] = temporal_disp

    return out_repre