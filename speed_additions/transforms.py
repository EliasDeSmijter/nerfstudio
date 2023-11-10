import numpy as np

def random_quaternion():
    z = 2.0
    w = 2.0
    while (z>1):
        x = np.random.uniform(-1.0,1.0)
        y = np.random.uniform(-1.0,1.0)
        z = x**2 + y**2
        
    while(w>1):
        u = np.random.uniform(-1.0,1.0)
        v = np.random.uniform(-1.0,1.0)
        w = u**2 + v**2
        
    s = np.sqrt((1-z) / w);
        
    return np.array([s*v, x, y, s*u])

def rotation2quaternion(M):
    tr = np.trace(M)
    m = M.reshape(-1)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[7] - m[5]) / s
        y = (m[2] - m[6]) / s
        z = (m[3] - m[1]) / s
    elif m[0] > m[4] and m[0] > m[8]:
        s = np.sqrt(1.0 + m[0] - m[4] - m[8]) * 2
        w = (m[7] - m[5]) / s
        x = 0.25 * s
        y = (m[1] + m[3]) / s
        z = (m[2] + m[6]) / s
    elif m[4] > m[8]:
        s = np.sqrt(1.0 + m[4] - m[0] - m[8]) * 2
        w = (m[2] - m[6]) / s
        x = (m[1] + m[3]) / s
        y = 0.25 * s
        z = (m[5] + m[7]) / s
    else:
        s = np.sqrt(1.0 + m[8] - m[0] - m[4]) * 2
        w = (m[3] - m[1]) / s
        x = (m[2] + m[6]) / s
        y = (m[5] + m[7]) / s
        z = 0.25 * s
    Q = np.array([w, x, y, z]).reshape(-1)
    return Q

def quaternion2rotation(quat):
    #print(quat)
    assert (len(quat) == 4)
    # normalize first
    quat = quat / np.linalg.norm(quat)
    a, b, c, d = quat

    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    ab = a * b
    ac = a * c
    ad = a * d
    bc = b * c
    bd = b * d
    cd = c * d

    # s = a2 + b2 + c2 + d2

    m0 = a2 + b2 - c2 - d2
    m1 = 2 * (bc - ad)
    m2 = 2 * (bd + ac)
    m3 = 2 * (bc + ad)
    m4 = a2 - b2 + c2 - d2
    m5 = 2 * (cd - ab)
    m6 = 2 * (bd - ac)
    m7 = 2 * (cd + ab)
    m8 = a2 - b2 - c2 + d2

    return np.array([m0, m1, m2, m3, m4, m5, m6, m7, m8]).reshape(3, 3)

def hamiltonian_product(q1, q2):
    # First Rotation q1,
    # Then, Rotation q2
    
    a1 = q1[0]
    b1 = q1[1]
    c1 = q1[2]
    d1 = q1[3]
    a2 = q2[0]
    b2 = q2[1]
    c2 = q2[2]
    d2 = q2[3]
    
    w = a1*a2 - b1*b2 - c1*c2 - d1*d2
    x = a1*b2 + b1*a2 + c1*d2 - d1*c2
    y = a1*c2 - b1*d2 + c1*a2 + d1*b2
    z = a1*d2 + b1*c2 - c1*b2 + d1*a2
    
    return np.array([w, x, y, z])

def quat_inv(q):    
    return np.array([-q[0], q[1], q[2], q[3]])

def angular_error(q, q_gt):
    eps = 1e-8
    rot_term = 2 * np.arccos(np.clip(q.reshape(1,-1) @ q_gt.reshape(-1,1), -1.0+eps, 1.0-eps))
    # Handle third and fourth quadrant
    rot_term_sup_pi = (rot_term >3.141592 ) *1.0
    rot_term = (2 * 3.141592 - rot_term) * rot_term_sup_pi + rot_term * (1.0 - rot_term_sup_pi)
    
    return rot_term
