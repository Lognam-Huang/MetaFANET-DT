"""
@description Contains functionality for controlling UAVs, Ground Users, 
Base Stations, and interfacing with Sionna (1.1.0) methods.
@start-date 11-8-2024
@updated 8-25-2025
@author(s) Everett Tucker
"""

import sionna
import math
import mitsuba as mi
import drjit as dr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple
from ortools.sat.python import cp_model
from sionna.rt import Transmitter, Receiver, PlanarArray, PathSolver, RadioMapSolver, RadioMaterialBase

# The Earth's gravitational acceleration in m/s^2
GRAVITATIONAL_ACCEL = 9.80665
# The Boltzmann Constant in Joules/Kelvin
BOLTZMANN_CONSTANT = 1.380649e-23
# Average Air Density Around the Area where the UAV Flies, kg / m^3
AIR_DENSITY = 1.213941
# The number of samples to use when calculating the energy consumption
NUM_INTEGRATION_SAMPLES = 1000


def _perfectMatching(matrix):
        """
        Utility function for assignLandmarks
        Determines if the graph represented by the adjacency matrix has a perfect bipartite matching

        Args:
            matrix (np.array(np.array(int)): The 0/1 adjacency matrix, should have shape=(n, n)
        
        Returns:
            Tuple(Boolean, np.array(int)): If a perfect matching is possible, followed by an optimal matching if possible, otherwise it is nonsense
        """
        n = len(matrix)
        match = [-1] * n  # Tracks which row is matched to each column
        graph = matrix    # The adjacency matrix
        
        def find_matching(u, seen):
            for v in range(n):
                if graph[u][v] == 1 and not seen[v]:
                    seen[v] = True
                    if match[v] == -1 or find_matching(match[v], seen):
                        match[v] = u
                        return True
            return False

        count = 0
        for u in range(n):
            seen = [False] * n
            if find_matching(u, seen):
                count += 1
                
        return count == n, match


"""
CustomRadioMaterial class written by Sionna to help with the integration of our own radio materials in the future
"""
class CustomRadioMaterial(RadioMaterialBase):

    # The __init__ method builds the radio material from:
    # - A unique `name` to identify the material instance in the scene
    # - The gain parameter `g`
    # - An optional `color` for displaying the material in the previewer and renderer
    # Providing these 3 parameters to __init__ is how an instance of this radio material
    # is built programmatically.
    #
    # When loading a scene from an XML file, Mitsuba provides to __init__
    # only an `mi.Properties` object containing all the properties of the material
    # read from the XML scene file. Therefore, when a `props` object is provided,
    # the other parameters are ignored and should not be given.
    def __init__(self,
                 name : str | None = None,
                 g : float | mi.Float | None = None,
                 color : Tuple[float, float, float] | None = None,
                 props : mi.Properties | None = None):

        # If `props` is `None`, then one is built from the
        # other parameters
        if props is None:
            props = mi.Properties("custom-radio-material")
            # Name of the radio material
            props.set_id(name)
            props["g"] = g
            if color is not None:
                props["color"] = mi.ScalarColor3f(color)

        # Read the gain from `props`
        g = 0.0
        if props.has_property("g"):
            g = props["g"]
            props.remove_property('g')
        self._g = mi.Float(g)

        # The other parameters (`name`, `color`) are given to the
        # base class to complete the initialization of the material
        super().__init__(props)

    def sample(
        self,
        ctx : mi.BSDFContext,
        si : mi.SurfaceInteraction3f,
        sample1 : mi.Float,
        sample2 : mi.Point2f,
        active : bool | mi.Bool = True
    ) -> Tuple[mi.BSDFSample3f, mi.Spectrum]:

        # Read the incident direction of propagation in the local coordinate
        # system
        ki_prime = si.wi

        # Build the 3x3 change-of-basis matrix from the local basis to the world
        # basis.
        # `si.sh_frame` stores the three vectors that define the local interaction basis
        # in the world coordinate system.
        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T

        # Direction of propagation of the reflected field in the local coordinate system
        kr_prime = mi.reflect(-ki_prime)

        # Compute the Jones matrix in the implicit world coordinate system
        # The `jones_matrix_to_world_implicit()` builds the Jones matrix with the
        # structure we need.
        sqrt_g = mi.Complex2f(dr.sqrt(self._g), 0.)
        jones_mat = sionna.rt.utils.jones_matrix_to_world_implicit(c1=sqrt_g,
                                                   c2=sqrt_g,
                                                   to_world=to_world,
                                                   k_in_local=ki_prime,
                                                   k_out_local=kr_prime)

        ## We now only need to prepare the outputs

        # Cast the Jones matrix to a `mi.Spectrum` to meet the requirements of
        # the BSDF interface of Mitsuba
        jones_mat = mi.Spectrum(jones_mat)

        # Instantiate and set the BSDFSample object
        bs = mi.BSDFSample3f()
        # Specifies the type of interaction that was computed
        bs.sampled_component = sionna.rt.constants.InteractionType.SPECULAR
        # Direction of the scattered wave in the world frame
        bs.wo = to_world@kr_prime
        # The next field of `bs` stores the probability that the sampled
        # interaction type and direction of scattering are sampled conditioned
        # on the given direction of incidence.
        # As only one event and direction of scattering are possible with this model,
        # this probability is set to 1.
        bs.pdf = mi.Float(1.)
        # Not used but required to be set
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DeltaReflection)
        bs.eta = 1.0

        return bs, jones_mat

    def eval(
        self,
        ctx : mi.BSDFContext,
        si : mi.SurfaceInteraction3f,
        wo : mi.Vector3f,
        active : bool | mi.Bool = True
    ) -> mi.Spectrum:

        # Read the incident direction of propagation in the local coordinate
        # system
        ki_prime = si.wi

        # Build the 3x3 change-of-basis matrix from the local basis to the world
        # basis.
        # `si.sh_frame` stores the three vectors that define the local interaction basis
        # in the world coordinate system.
        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T

        # Direction of propagation of the reflected field in the local coordinate system
        kr_prime = mi.reflect(-ki_prime)

        # Compute the Jones matrix in the implicit world coordinate system
        # The `jones_matrix_to_world_implicit()` builds the Jones matrix with the
        # structure we need.
        sqrt_g = mi.Complex2f(dr.sqrt(self._g), 0.)
        jones_mat = sionna.rt.utils.jones_matrix_to_world_implicit(c1=sqrt_g,
                                                   c2=sqrt_g,
                                                   to_world=to_world,
                                                   k_in_local=ki_prime,
                                                   k_out_local=kr_prime)

        # This model only scatters energy in the direction of the specular reflection.
        # Any other direction provided by the user `wo` should therefore lead to no energy.
        is_valid = sionna.rt.utils.isclose(dr.dot(kr_prime, wo), mi.Float(1.))
        jones_mat = dr.select(is_valid, jones_mat, 0.)

        # Cast the Jones matrix to a `mi.Spectrum` to meet the requirements of
        # the BSDF interface of Mitsuba
        jones_mat = mi.Spectrum(jones_mat)

        return jones_mat

    def pdf(
        self,
        ctx : mi.BSDFContext,
        si : mi.SurfaceInteraction3f,
        wo : mi.Vector3f,
        active : bool | mi.Bool = True
    ) -> mi.Float:

        # Read the incident direction of propagation in the local coordinate
        # system
        ki_prime = si.wi

        # Direction of propagation of the reflected field in the local coordinate system
        kr_prime = mi.reflect(-ki_prime)

        # As only one event and direction of scattering are possible with this model,
        # the probability is set to 1 for this direction and 0 for any other.
        is_valid = sionna.rt.utils.isclose(dr.dot(kr_prime, wo), mi.Float(1.))
        return dr.select(is_valid, mi.Float(1.), mi.Float(0.))

    def traverse(self, callback : mi.TraversalCallback):
        # Registers the `g` parameter as a differentiable
        # parameter of the scene
        callback.put_parameter('g', self._g,
                               mi.ParamFlags.Differentiable)

    def to_string(self) -> str:
        # Returns a humanly readable description of the material
        s = f"CustomRadioMaterial["\
            f"g={self._g}"\
            f"]"
        return s

    # We add a getter and setter to access `g`
    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, v):
        self._g = mi.Float(v)

"""
TODO:
Change the path loss functions or introduce another function that
calculates the maximum achievable data rate over the connection.
That requires having the signal-to-noise ratio, the initial signal
power, and the path loss. I already have the path loss through
so I just have to worry about the other stuff.

Perhaps the SNR is not dependent on the path itself though and could
be thought of as constant, as a hyperparameter of the simulation.
"""
class Environment():
    def __init__(self, scene_path, position_df_path, desired_throughputs=None, time_step=1, ped_height=1.5, ped_rx=True, ped_color=np.array([0, 1, 0]), wind_vector=np.zeros(3), temperature=290):
        """
        Creates a new environment from a scene path and a position_df_path
        This method may take several minutes to run because of the scene creation

        Args:
            scene_path (str): the file path of the XML scene from Mitsuba
            position_df_pat (str): the file path of the csv position data
            time_step (float): the time step between simulation iterations
            ped_height (float): the assumed height of the pedestrians, in meters
            ped_rx (bool): if the pedestrians are receivers and the UAVs are transmitters, default true
            ped_color (np.array(3,)): The RGB Color of the pedestrians, where each entry is in [0, 1]. Defaults to black.
            wind_vector (np.array(3,)): The velocity of the wind in the environment
            temperature (float): the temperature of the environment, in degrees Kelvin
        """
        
        # Registering custom materials with Mitsuba
        mi.register_bsdf("diffuse", lambda props: CustomRadioMaterial(props=props))
        mi.register_bsdf("twosided", lambda props: CustomRadioMaterial(props=props))
        mi.register_bsdf("principle", lambda props: CustomRadioMaterial(props=props))

        # Creating the scene and initializing environment variables
        self.scene = sionna.rt.load_scene(scene_path, merge_shapes=False)
        self.ped_rx = ped_rx
        self.time_step = time_step
        self.cur_step = 0
        self.ped_height = ped_height
        self.ped_color = ped_color  # The default is green like in the Sionna visualizations
        self.n_rx = 0
        self.n_tx = 0
        self.n_bs = 0
        self.uavs = []
        self.base_stations = []
        # self.gus = self.createGroundUsers(position_df_path, desired_throughputs)
        # try to get GUs from training
        if position_df_path is not None:
            self.gus = self.createGroundUsers(position_df_path, desired_throughputs)
        else:
            self.gus = np.array([])
        self.temperature = temperature
        self.wind = wind_vector
        # This is used to speed up computation later in the simulation
        self.bezier_matrix = np.linalg.inv(np.array([
            [1, 0, 0, 0], 
            [-3, 3, 0, 0], 
            [(1 - self.time_step) ** 3, 3 * self.time_step * (1 - self.time_step) ** 2, 3 * self.time_step ** 2 * (1 - self.time_step), self.time_step ** 3], 
            [-3 * (1 - self.time_step) ** 2, -6 * self.time_step * (1 - self.time_step) + 3 * (1 - self.time_step) ** 2, -3 * self.time_step ** 2 + 6 * self.time_step * (1 - self.time_step), 3 * self.time_step ** 2]
        ]))

        # Save scene bounding box as (lower, upper)
        mi_bbox = self.scene.mi_scene.bbox()
        self.scene_bbox = (mi_bbox.min.numpy(), mi_bbox.max.numpy())



    def createGroundUsers(self, position_df_path, desired_throughputs=None):
        """
        Parses the positions data from SUMO and creates a list of ground user objects
        Also initializes the inital positions with receivers or transmitters

        Args:
            position_df_path (str): the file path to the pedestrian position dataframe
            desired_throughputs (np.array(num_gus, num_time_steps)): the desired throughput time series for each ground user, in bits per second
        
        Returns:
            np.array(): a list of ground user objects
        """

        df = pd.read_csv(position_df_path)
        i = 0
        res = []
        while (len(df.loc[df["person_id"] == "ped" + str(i)]) > 0):
            res.append(df.loc[df["person_id"] == "ped" + str(i)])
            i += 1
        
        # Creating the ground users
        rtn = []
        for j in range(len(res)):
            if desired_throughputs is None:
                rtn.append(GroundUser(j, np.array([res[j]["local_person_x"], res[j]["local_person_y"], np.full(len(res[j]), 
                                      self.ped_height)]).T, height=self.ped_height, com_type=("rx" if self.ped_rx else "tx"), 
                                      delta_t=self.time_step, color=self.ped_color))
            else:
                rtn.append(GroundUser(j, np.array([res[j]["local_person_x"], res[j]["local_person_y"], np.full(len(res[j]), 
                                      self.ped_height)]).T, height=self.ped_height, com_type=("rx" if self.ped_rx else "tx"), 
                                      delta_t=self.time_step, color=self.ped_color, desired_throughputs=desired_throughputs[j]))
            if self.ped_rx:
                self.n_rx += 1
            else:
                self.n_tx += 1
            self.scene.add(rtn[j].device)
                
        return np.array(rtn)
    
    def addGU(self, pos, height=1.5, com_type="rx", color=np.array([0, 1, 0]), delta_t=1, desired_throughputs=None):
        """
        Adds a Ground User (GU) to the environment manually.
    
        Args:
            pos (np.array(3,)): Position of GU (x, y, z)
            height (float): GU device height, default 1.5m
            com_type (str): "rx" or "tx"
            color (np.array(3,)): RGB color for visualization
            delta_t (float): timestep size
            desired_throughputs (np.array): optional throughput series
        """
        gu_id = len(self.gus)
    
        # Stationary GU has only one position
        positions = np.array([pos])
    
        gu = GroundUser(
            id=gu_id,
            positions=positions,
            height=height,
            com_type=com_type,
            delta_t=delta_t,
            color=color,
            desired_throughputs=desired_throughputs if desired_throughputs is not None else np.full(150, 375000)
        )
    
        # Register GU
        self.gus = np.append(self.gus, gu)
        if com_type == "rx":
            self.n_rx += 1
        else:
            self.n_tx += 1
        self.scene.add(gu.device)
    
        return gu_id

    def advancePedestrianPositions(self):
        """
        Advances all the pedestrian positions to the next their next time step
        """

        for id in range(len(self.gus)):
            self.updateGroundUser(id)
    

    def step(self, uav_params):
        """
        Updates the simulation by moving all ground users to the next position,
        updating the UAVs' signal power, and moving all the uavs to the positions
         specified in UAV params

        Args:
            dict(int, (float, np.array(3,), np.array(3,))): A dictionary of signal power, position, velocity tuples
            that describe the new states of all the UAVs. Any uavs without a key in
            uav_positions will remain in place after the step and their signal power will remain constant.
        """

        self.cur_step += 1
        for x in range(len(self.uavs)):
            if x in uav_params:
                self.moveAbsUAV(x, uav_params[x][1], uav_params[x][2])
            else:
                self.moveAbsUAV(x, self.uavs[x].pos, self.uavs[x].vel)
            self.setUAVSignalPower(x, uav_params[x][0])
        
        for bs in self.base_stations:
            bs.computeConsumption()

        self.advancePedestrianPositions()


    def computeShortestDistance(self):
        """
        Computes the shortest distance between any two UAVs, used for collision avoidance
        I would use the O(nlogn) algorithm, but I think the constant time factor is way too high for our application
        This algorithm is better when len(self.uavs) < 125, which I think is basically always guarenteed

        Returns:
            the shortest squared distance between any two uavs, in m
        """
        rtn = math.inf
        for i in range(len(self.uavs)):
            for j in range(i + 1, len(self.uavs)):
                rtn = min(rtn, 
                (self.uavs[i].pos[0] - self.uavs[j].pos[0]) ** 2 +
                (self.uavs[i].pos[1] - self.uavs[j].pos[1]) ** 2 +
                (self.uavs[i].pos[2] - self.uavs[j].pos[2]) ** 2)

        return math.sqrt(rtn)
            

    def visualize(self, paths=None, radio_map=None):
        """
        Visualizes the current receivers and transmitters in the scene.
        Includes the paths and/or radio map, if provided.
        Uses the SINR metric for the radio maps by default

        Args:
            (sionna.rt.Paths): A paths object that you want to display in simulation
            (sionna.rt.RadioMap): A radio map object that you want to display
        """

        if paths is None:
            if radio_map is None:
                self.scene.preview(show_devices=True)
            else:
                self.scene.preview(show_devices=True, radio_map=radio_map, rm_metric="sinr")
        else:
            if radio_map is None:
                self.scene.preview(show_devices=True, paths=paths)
            else:
                self.scene.preview(show_devices=True, paths=paths, radio_map=radio_map, rm_metric="sinr")
    

    def computeRadioMap(self, max_depth=2, num_samples=1000000, cell_size=(1.0, 1.0)):
        """
        Computes a radio map for every transmitter using the provided
        arguments for maximum reflection depth and the number of samples
        on the fibonacci sphere.

        Args:
            max_depth (int): the maximum reflection depth computed
            num_samples (int): the number of points to sample for each transmitter in the scene
            cell_size ([int, int]): The size of the radio map chuncks. Works like resolution, in meters
        """
        solver = RadioMapSolver()
        return solver(self.scene, cell_size=cell_size, samples_per_tx=num_samples, max_depth=max_depth, 
                      los=True, specular_reflection=True, diffuse_reflection=True, refraction=True)
    

    def getUserSINRS(self, radio_map):
        """
        Computes the SINR values for each Ground User based on their position in the specified radio map
        Currently, these are just merged linearly, but they should probably use diversity combining to merge signals from each UAV

        Args:
            radio_map (sionna.rt.PlanarRadioMap): The radio map to use for computing the SINR, can be obtained with computeRadioMap
        
        Returns:
            np.array(np.array(float)): An array of the SINR values for each Ground User to each UAV, shape=(num_rx,num_tx)
        """

        # Indices are in format (column, row)
        indices = radio_map.rx_cell_indices.numpy().T
        rtn = []
        for i in range(self.n_rx):
            rtn.append(radio_map.sinr[:, int(indices[i][1]), int(indices[i][0])])

        return np.array(rtn)
    

    def computeAlpha(self, max_depth, num_samples):
        """
        Computes the path coefficients for all paths between each pair of receivers and transmitters
        
        Args:
            max_depth (int): the maximum reflection depth computed, 2 is standard
            num_samples (int): the number of points to sample on the fibonacci sphere, 10^4 or 10^5 works well

        Returns:
            tf.tensor(num_rx, num_tx, max_num_paths): The channel impulse responses for each receiver and transmitter pair
        """

        paths = self.computeGeneralPaths(max_depth, num_samples)

        # Check the sampling frequency parameter for the doppler shift
        if self.ped_rx:
            paths.apply_doppler(0.0001, 1, np.array([x.vel + self.wind for x in self.uavs]), np.array([x.getVelocity() for x in self.gus]))
        else:
            paths.apply_doppler(0.0001, 1, np.array([x.getVelocity() for x in self.gus]), np.array([x.vel + self.wind for x in self.uavs]))
        
        a, tau = paths.cir(los=True, reflection=True, diffraction=True, scattering=True, ris=False)

        return tf.squeeze(a).numpy().astype(np.float64)


    def computeLOSPaths(self, mode='gpu'):
        """
        Computes the line-of-sight paths for all potential receivers and transmitters.
        Requires GPU support.

        Args:
            mode (str ['gpu', 'cpu']) the type of hardware you are using to run Sionna, can be checked with
            tf.config.list_physical_devices('GPU')

        Returns:
            (sionna.rt.Paths): All possible line-of-sight paths
        """
        dr.flush_malloc_cache()  # Freeing up memory, because this uses a lot

        # Setting the mitsuba variant depending on the computation mode
        if mode == 'gpu':
            mi.set_variant("cuda_ad_mono_polarized")
        elif mode == 'cpu':
            mi.set_variant("llvm_ad_mono_polarized")
        else:
            raise TypeError(f"Mode should be 'cpu' or 'gpu' not {mode}")

        solver = PathSolver()
        paths = solver(self.scene, max_depth=1, max_num_paths_per_src=self.n_rx, samples_per_src=1, synthetic_array=True,
                      los=True, specular_reflection=False, diffuse_reflection=False, refraction=False)
        paths.interactions[:, :, :, :] = 0  # Line to fix a weird error, all should be type=0 for LoS
        return paths


    def computeLOSDataRate(self):
        """
        DEPRECATED IN Sionna 1.1.0
        Computes the average Line-of-sight theoretical maximum data rate across all pairs of transmitters and receivers.
        A larger value means that the paths can send more data and the UAVs are in more
        optimal positions for communication with the Ground Users.

        Returns:
            np.array(num_rx, num_tx): The total theoretical maximum data rate for each pair of receivers and transmitters across all line-of-sight paths in the simulation
        """

        raise NotImplementedError("Function is deprecated in Sionna 1.1.0")
        """
        paths = self.computeLOSPaths()

        # Check the sampling frequency parameter for the doppler shift
        if self.ped_rx:
            paths.apply_doppler(0.0001, 1, np.array([x.vel + self.wind for x in self.uavs]), np.array([x.getVelocity() for x in self.gus]))
        else:
            paths.apply_doppler(0.0001, 1, np.array([x.getVelocity() for x in self.gus]), np.array([x.vel + self.wind for x in self.uavs]))
        
        a, tau = paths.cir(los=True, reflection=False, diffraction=False, scattering=False, ris=False)
        
        # Computes the sum of the theoetical maximum data rates for each UAV in simulation
        # r_max = Blog2(1 + (Pt * a^2) / kTB); B = bandwidth (Mbps), Pt = transmission power (W), a = path coefficients (unitless), k = Boltzmann Constant (J/K), T = temperature (Kelvin)

        a = tf.squeeze(a)  # TODO: If I only have one User or UAV in the simulation this is going to cause an error
        bandwidth = tf.convert_to_tensor([uav.bandwidth for uav in self.uavs], dtype=tf.float32)
        bandwidth = tf.broadcast_to(tf.reshape(bandwidth, [1, -1]), [self.n_rx, self.n_tx])
        signal_power = tf.convert_to_tensor([uav.signal_power for uav in self.uavs], dtype=tf.float32)
        signal_power = tf.broadcast_to(tf.reshape(signal_power, [1, -1]), [self.n_rx, self.n_tx])

        return bandwidth * np.log2(1 + (signal_power * np.abs(a) ** 2) / (BOLTZMANN_CONSTANT * self.temperature * bandwidth))
        """
    

    def computeGeneralPaths(self, max_depth, num_samples, mode='gpu'):
        """
        Computes line-of-sight, reflection, diffraction, and scattering
        paths from all transmitters. Requires GPU support

        Args:
            max_depth (int): the maximum reflection depth usually 2-3 works well
            num_samples (int): the number of sample points to take from the
            fiboncci sphere, usually about 10^4 or 10^5
            mode (str ['gpu', 'cpu']) the type of hardware you are using to run Sionna, can be checked with
            tf.config.list_physical_devices('GPU')
        
        Returns:
            (sionna.rt.Paths): All possible paths in the environment
        """
        dr.flush_malloc_cache()  # Freeing up memory, because this uses a lot

        # Setting the mitsuba variant depending on the computation mode
        if mode == 'gpu':
            mi.set_variant("cuda_ad_mono_polarized")
        elif mode == 'cpu':
            mi.set_variant("llvm_ad_mono_polarized")
        else:
            raise TypeError(f"Mode should be 'cpu' or 'gpu' not {mode}")
        
        solver = PathSolver()
        return solver(self.scene, max_depth=max_depth, max_num_paths_per_src=num_samples, 
                      samples_per_src=num_samples, los=True, specular_reflection=True, 
                      diffuse_reflection=True, refraction=True)


    def computeGeneralDataRate(self, max_depth, num_samples):
        """
        Computes the average theoretical maximum data rate for all different types of paths, including
        line-of-sight, reflection, diffraction, and scattering for each transmitter. A
        larger value means that a UAV is in a more optimial position and can send
        more data to/from the Ground Users.

        Args:
            max_depth (int): the maximum reflection depth usually 2-3 works well
            num_samples (int): the number of sample points to take from the
            fiboncci sphere, usually about 10^4 or 10^5
        
        Returns:
            np.array(num_rx, num_tx): The theoretical maximum data rate of all possible paths for each transmitter
        """

        paths = self.computeGeneralPaths(max_depth, num_samples)

        # Check the sampling frequency parameter for the doppler shift
        if self.ped_rx:
            paths.apply_doppler(0.0001, 1, np.array([x.vel + self.wind for x in self.uavs]), np.array([x.getVelocity() for x in self.gus]))
        else:
            paths.apply_doppler(0.0001, 1, np.array([x.getVelocity() for x in self.gus]), np.array([x.vel + self.wind for x in self.uavs]))
        
        a, tau = paths.cir(sampling_frequency=1.0, num_time_steps=num_samples, normalize_delays=False, reverse_direction=False, out_type='tf')
        
        # Computes the sum of the theoetical maximum data rates for each UAV in simulation
        # r_max = Blog2(1 + (Pt * a^2) / kTB); B = bandwidth (Mbps), Pt = transmission power (W), a = path coefficients (unitless), k = Boltzmann Constant (J/K), T = temperature (Kelvin)

        a = np.abs(tf.squeeze(a))
        bandwidth = tf.convert_to_tensor([uav.bandwidth for uav in self.uavs], dtype=tf.float32)
        bandwidth = tf.broadcast_to(tf.reshape(bandwidth, [1, -1, 1]), [self.n_rx, self.n_tx, tf.shape(a)[2]])
        signal_power = tf.convert_to_tensor([uav.signal_power for uav in self.uavs], dtype=tf.float32)
        signal_power = tf.broadcast_to(tf.reshape(signal_power, [1, -1, 1]), [self.n_rx, self.n_tx, tf.shape(a)[2]])

        return tf.math.reduce_sum(bandwidth * np.log2(1 + (signal_power * a ** 2) / (BOLTZMANN_CONSTANT * self.temperature * bandwidth)), axis=2).numpy().astype(np.float64)
    

    def addUAV(self, mass=1, efficiency=0.8, pos=np.zeros(3), vel=np.zeros(3), color=np.array([1, 0, 0]), bandwidth=50, rotor_area=None, signal_power=0, throughput_capacity=625000000, battery_capacity=10000):
        """
        Adds a UAV to the environment and initalizes its quantities and receiver / transmitter

        Args:
            mass (float): the mass of the UAV in kilograms
            efficiency (float): the proportion of the UAV's power consumption that is turned into movement
            pos (np.array(3,)): the initial position of the UAV
            vel (np.array(3,)): the initial velocity of the UAV2
            color (np.array(3,)): the color of the UAV used for visualization
            bandwidth (float): the bandwidth of the UAV's communication channel, in Mbps
            rotor_area (float): the area of the UAV's rotors, in m^2
            signal_power (float): the transmitter/receiver power of the UAV, in watts
            throughput_capacity (float): the maximum throughput of the Base Station in Bytes per second, default 625,000,000
            battery_capacity (float): the battery capacity of the UAV, in Joules
        
        Returns:
            (int) the id of the created UAV
        """

        id = self.n_tx if self.ped_rx else self.n_rx
        if rotor_area is None:
            self.uavs.append(UAV(id, mass, efficiency, pos, vel - self.wind, bandwidth, self.time_step, mass * 0.3, signal_power, throughput_capacity, battery_capacity))
        else:
            self.uavs.append(UAV(id, mass, efficiency, pos, vel - self.wind, bandwidth, self.time_step, rotor_area, signal_power, throughput_capacity, battery_capacity))

        if self.ped_rx:
            self.uavs[id].device = Transmitter(name=f"uav{id}", position=mi.Point3f(pos), color=color)
            self.n_tx += 1
        else:
            self.uavs[id].device = Receiver(name=f"uav{id}", position=mi.Point3f(pos), color=color)
            self.n_rx += 1
        
        self.uavs[id].lookAt()  # Pointing the UAV at the ground to start
        self.scene.add(self.uavs[id].device)
        return len(self.uavs) - 1


    def addBaseStation(self, device_type="tx", pos=np.zeros(3,), color=np.array([1, 0, 0]), bandwidth=50, signal_power=0, throughput_capacity=625000000, battery_capacity=10000):
        """
        Adds a new Base Station to the simulation with the specified parameters

        Args:
            pos (np.array(3,)): the initial position of the UAV
            device_type (str): the type of the device associated with the Base Station, either "tx" or "rx" default "tx"
            color (np.array(3,)): the color of the UAV used for visualization
            bandwidth (float): the bandwidth of the UAV's communication channel, in Mbps
            signal_power (float): the transmitter/receiver power of the UAV, in watts
            throughput_capacity (float): the maximum throughput of the Base Station in Bytes per second, default 625,000,000
        
        Returns:
            (int): the unique id of the Base Station
        """
        assert device_type in ["rx", "tx"]

        id = self.n_bs
        self.base_stations.append(BaseStation(id, pos, bandwidth, self.time_step, signal_power, throughput_capacity, battery_capacity))
        if device_type == "tx":
            self.base_stations[id].device = Transmitter(name=f"bs{id}", position=mi.Point3f(pos), color=color)
        else:
            self.base_stations[id].device = Receiver(name=f"bs{id}", position=mi.Point3f(pos), color=color)
        self.n_bs += 1
        
        self.base_stations[id].lookAt()  # Pointing the Base Station at the ground to start
        self.scene.add(self.base_stations[id].device)
        return self.n_bs - 1


    def setUAVSignalPower(self, id, power):
        """
        Updates the signal power of the specified UAV

        Args:
            id (int): the unique id of the UAV
            power (float): the signal power of the UAV, in watts
        """
        if power >= 0:
            self.uavs[id].signal_power = power
        else:
            raise ValueError("Signal power must be non-negative")


    def moveAbsUAV(self, id, abs_pos, abs_vel):
        """
        Moves the uav with the specified id to a a new absolution position and velocity
        Also updates the position of the communication device associated with the UAV

        Args:
            id (int): the unique id of the UAV
            abs_pos (np.array(3,)): the absolute position vector of the UAV after the move
            abs_vel (np.array(3,)): the absolute velocity vector of the UAV after the move
        """
        self.uavs[id].move(abs_pos, abs_vel - self.wind, self.bezier_matrix)
        self.uavs[id].device.position = abs_pos


    def moveRelUAV(self, id, relative_pos):
        """
        Moves the UAV to a new position relative to its current position, maintains current velocity
        Also updates the position of the communication device associated with the UAV

        Args:
            id (int): the unique id of the UAV
            relative_pos (np.array(3,)): the relative position of the UAV
        """
        self.uavs[id].move(self.uavs[id].pos + relative_pos, self.uavs[id].vel, self.bezier_matrix)
        self.uavs[id].device.position = self.uavs[id].pos + relative_pos


    def getUAVPos(self, id):
        """
        Gets the UAV position by Id

        Args:
            id (int): the unique id of the UAV

        Returns:
            np.array(3,): the position vector of the UAV
        """
        return self.uavs[id].pos


    def getUAVVel(self, id):
        """
        Gets the absolute velocity of the UAV by Id

        Args:
            id (int): the unique id of the UAV
        
        Returns:
            np.array(3,): the absolute velocity of the UAV
        """
        return self.uavs[id].vel + self.wind


    def getConsumptions(self):
        """
        Returns a dictionary of all the id : consumption pairs for each Base Station or UAV
        currently in the simulation

        Returns:
            dict(int, float): A dictionary of all the id, consumption pairs for all uavs
        """

        rtn = {}
        for id in range(len(self.uavs)):
            rtn[id] = self.uavs[id].getConsumption()
        return rtn


    def updateGroundUser(self, id):
        """
        Moves a ground user to the next avaliable position and updates velocity
        """
        self.gus[id].update()
        # Just update the x and y positions, the height stays constant
        self.gus[id].device.position = tf.constant([self.gus[id].getCurrentPosition()[0], self.gus[id].getCurrentPosition()[1], self.gus[id].height])
    

    def setTransmitterArray(self, arr=None):
        """
        Sets the scene's transmitter array to arr, sets to a default isotropic antenna if arr is None

        Args:
            arr the antenna array for the transmitters
        """
        if arr is None:
            # Initializing a single isotropic antenna
            self.scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
        else:
            self.scene.tx_array = arr


    def setReceiverArray(self, arr=None):
        """
        Sets the scene's receiver array to arr, sets to a default isotropic antenna if arr is None

        Args:
            arr the antenna array for the receivers
        """
        if arr is None:
            self.scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
        else:
            self.scene.rx_array = arr


    def plotBS(self, length=None):
        """
        Plots the positions and velocities of all the current Base Stations in 3 Dimensions

        Args:
            length (float): the length of the normalized velocity vectors, None if not normalized
        """
        axis = plt.figure().add_subplot(projection='3d')
        for uav in self.uavs:
            c = np.random.rand(3)
            axis.scatter(uav.pos[0], uav.pos[1], uav.pos[2], color=c)
            if not np.array_equal(uav.vel, np.zeros(3)):
                x, y, z = np.meshgrid(uav.pos[0], uav.pos[1], uav.pos[2])
                if length is None:
                    axis.quiver(x, y, z, uav.vel[0], uav.vel[1], uav.vel[2], color=c)
                else:
                    axis.quiver(x, y, z, uav.vel[0], uav.vel[1], uav.vel[2], length=length, normalize=True, color=c)
        plt.show()


    def plotGUs(self, timestep=None):
        """
        Plots the Ground Users in 2D at the given timestep

        Args:
            timestep (int): The time to get the position of the Ground User at, default is current step
        """

        if timestep is None:
            timestep = self.cur_step
        for gu in self.gus:
            plt.scatter(gu.getTimestampPosition(timestep)[0], gu.getTimestampPosition(timestep)[1])
        plt.title(f'Ground User Positions for timestep: {timestep}')
        plt.xlabel("x-coordinate")
        plt.ylabel('y-coordinate')
        plt.show()


    def getBezier(self, x_i, x_f, v_i, v_f, samples):
        """
        Returns a number of sample points along the Bezier curve from the given position-velocity pairs

        Args:
            x_i (np.array(3,)): The initial position vector, in meters
            x_f (np.array(3,)): The final position vector, in meters
            v_i (np.array(3,)): The initial velocity vector, in m/s
            v_f (np.array(3,)): The final velocity vector, in m/s
        
        Returns:
            np.array(samples,3): An array of points along the calculated Bezier curve
        """

        f = np.array([x_i, v_i, x_f, v_f])
        b = np.dot(self.bezier_matrix, f)

        rtn = []
        for i in range(samples):
            t = self.time_step * i / samples
            # Add the value of the parametric curve at time t
            rtn.append(((1 - t) ** 3) * b[0] + 3 * t * ((1 - t) ** 2) * b[1] + 3 * (t ** 2) * (1 - t) * b[2] + (t ** 3) * b[3])

        return np.array(rtn)
    

    # TODO: Delete, likely deprecated
    def getDataRateFromAssignments(self, assignments, path_qualities):
        """
        Computes the data rate for each UAV with the given assignments and path qualities, this is the theoretical, not actual

        Args:
            assignments (list(list(int))): assignments[i] contains a list of GUs assigned to UAV i.
            path_qualities (np.array(int)): Theoretical maximum throughput value between each ground user and UAV.

        Returns:
            np.array(num_tx + 1): The data rates of each transmitter, plus the sum of all the data rates  
        """

        rtn = np.zeros(self.n_tx + 1, dtype=np.int64)

        for i in range(self.n_tx):
            for j in assignments[i]:
                rtn[i] += path_qualities[j][i]
        rtn[-1] = np.sum(rtn)  # The last value is zero, so we just sum all of them
        return rtn


    def assignGUs(self, path_qualities):
        """
        Assigns the ground users by prioritizing the total throughput between all UAV-GU connections and ensuring that each
        ground user is assigned to at most one UAV. We assume that the UAV data rate exactly matches the sum of the desired
        throughputs of their assigned ground users. Then we backcalculate signal power to make it work out.

        Args:
            path_qualities (np.array(int)): Theoretical maximum throughput value between each ground user and UAV.
        
        Returns:
            assignments (list(list(int))): assignments[i] contains the list of GUs assigned to UAV i.
            data_rates (list(int)): The data rates of each transmitter, plus the sum of all the data rates. Has length n_tx + 1.
        """

        # Creating data arrays from environment data
        capacities = np.array([x.throughput_capacity for x in self.uavs], dtype=np.int64)
        desired_throughputs = np.array([x.getDesiredThroughput() for x in self.gus], dtype=np.int64)

        # Ensuring path quality values are integers, necessary for the solver
        try:
            path_qualities = path_qualities.astype(np.int64)
        except ValueError:
            raise ValueError("Input data must be of integer type or broadcastable.")

        model = cp_model.CpModel()
        gus = range(self.n_rx)
        uavs = range(self.n_tx)
        
        x = []
        for i in gus:
            x.append([])
            for j in uavs:
                x[i].append(model.NewBoolVar(""))
        x = np.array(x)

        # Assigning each gu to at most one uav
        for i in gus:
            model.Add(np.sum(x[i, :]) <= 1)

        # Ensuring each uav's assigned gus do not exceed its total capacity
        for j in uavs:
            model.Add(np.dot(desired_throughputs, x[:, j]) <= capacities[j])

        # Setting the objective and solving
        model.Maximize(np.sum(path_qualities * x))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # If the model failed to solve the problem
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise ValueError("Input is unsolvable or infeasible.")

        # Getting uav assignments
        assignments = []
        for j in uavs:
            assignments.append([])
            for i in gus:
                if solver.Value(x[i][j]):
                    assignments[j].append(i)

        # Generating the data rates for each UAV, taking the min of path qualities and desired_throughput
        data_rates = np.zeros(self.n_tx + 1, dtype=np.int64)
        for i in range(self.n_tx):
            for j in assignments[i]:
                data_rates[i] += min(desired_throughputs[j], path_qualities[j][i])
        data_rates[-1] = np.sum(data_rates[:-1])

        return assignments, data_rates


    def assignGUsByCount(self, path_qualities, alpha=1, beta=1):
        """
        Assigns the ground users by maximizing the total throughput objective as well as load balancing by the number of GUs assigned
        to each UAV.

        Args:
            path_qualities (np.array(int)): Theoretical maximum throughput value between each ground user and UAV.
            alpha (float): Optimization coefficient for the throughput maximization objective.
            beta (float): Optimization coefficient for the minimum number of GUs per UAV objective.
        
        Returns:
            assignments (list(list(int))): assignments[i] contains the list of GUs assigned to UAV i.
            total_throughput (int): The total theoretical maximum throughput of all UAV-GU connections.
        """

        # Creating data arrays from environment data
        capacities = np.array([x.throughput_capacity for x in self.uavs], dtype=np.int64)
        desired_throughputs = np.array([x.getDesiredThroughput() for x in self.gus], dtype=np.int64)

        # Ensuring path quality values are integers, necessary for the solver
        try:
            path_qualities = path_qualities.astype(np.int64)
        except ValueError:
            raise ValueError("Input data must be of integer type or broadcastable.")

        model = cp_model.CpModel()
        gus = range(self.n_rx)
        uavs = range(self.n_tx)
        
        x = []
        for i in gus:
            x.append([])
            for j in uavs:
                x[i].append(model.NewBoolVar(""))
        x = np.array(x)

        # Assigning each gu to at most one uav
        for i in gus:
            model.Add(np.sum(x[i, :]) <= 1)

        # Ensuring each uav's assigned gus do not exceed its total capacity
        for j in uavs:
            model.Add(np.dot(desired_throughputs, x[:, j]) <= capacities[j])

        # Defining the task counts for each uav
        assignment_counts = np.array([model.NewIntVar(0, self.n_rx, "") for j in uavs])
        for j in uavs:
            model.Add(assignment_counts[j] == np.sum(x[:, j]))

        # Defining the minimum number of gus across all uavs
        min_gus = model.NewIntVar(0, self.n_rx, "")
        for j in uavs:
            model.Add(assignment_counts[j] >= min_gus)

        # Maximizing the combined objective
        model.Maximize(np.sum(path_qualities * x) * alpha + min_gus * beta)

        # Solving
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # If the model failed to solve the problem
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise ValueError("Input is unsolvable or infeasible.")

        # Getting uav assignments
        assignments = []
        for j in uavs:
            assignments.append([])
            for i in gus:
                if solver.Value(x[i][j]):
                    assignments[j].append(i)

        ## Generating the data rates for each UAV, taking the min of path qualities and desired_throughput
        data_rates = np.zeros(self.n_tx + 1, dtype=np.int64)
        for i in range(self.n_tx):
            for j in assignments[i]:
                data_rates[i] += min(desired_throughputs[j], path_qualities[j][i])
        data_rates[-1] = np.sum(data_rates[:-1])

        return assignments, data_rates

    
    def assignGUsByThroughputWaste(self, path_qualities, scale=1, alpha=1, beta=1):
        """
        Args:
            path_qualities (np.array(int)): Theoretical maximum throughput value between each ground user and UAV.
            scale (int): scales down the input data to make the computation more efficient, should be positive
            alpha (float): Optimization coefficient for the throughput maximization objective, should be positive.
            beta (float): Optimization coefficient for the maximum throughput waste minimization objective, should be positive.
        
        Returns:
            assignments (list(list(int))): assignments[i] contains a list of GUs assigned to UAV i.
            total_throughput (int): The total theoretical maximum throughput of all UAV-GU connections.
        """

        # Creating data arrays from environment data
        capacities = np.array([x.throughput_capacity for x in self.uavs], dtype=np.int64)
        desired_throughputs = np.array([x.getDesiredThroughput() for x in self.gus], dtype=np.int64)

        # Ensuring path quality values are integers, necessary for the solver
        try:
            path_qualities = path_qualities.astype(np.int64)
        except ValueError:
            raise ValueError("Input data must be of integer type or broadcastable.")
        
        if scale > 1:
            path_qualities = np.floor(path_qualities / scale).astype(np.int64)
            capacities = np.floor(capacities / scale).astype(np.int64)
            desired_throughputs = np.floor(desired_throughputs / scale).astype(np.int64)

        model = cp_model.CpModel()
        gus = range(self.n_rx)
        uavs = range(self.n_tx)
        
        x = []
        for i in gus:
            x.append([])
            for j in uavs:
                x[i].append(model.NewBoolVar(""))
        x = np.array(x)

        # Assigning each gu to at most one uav
        for i in gus:
            model.Add(np.sum(x[i, :]) <= 1)

        # Ensuring each uav's assigned gus do not exceed its total capacity
        for j in uavs:
            model.Add(np.dot(desired_throughputs, x[:, j]) <= capacities[j])

        # Defining the task counts for each uav
        throughput_wastes = np.array([model.NewIntVar(0, capacities[j], "") for j in uavs])
        for j in uavs:
            model.Add(throughput_wastes[j] == capacities[j] - np.dot(desired_throughputs, x[:, j]))

        # Defining the minimum number of gus across all uavs
        max_throughput_waste = model.NewIntVar(0, np.max(capacities), "")
        for j in uavs:
            model.Add(throughput_wastes[j] <= max_throughput_waste)

        # Maximizing the combined objective
        model.Maximize(np.sum(path_qualities * x) * alpha - max_throughput_waste * beta)

        # Solving
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        # If the model failed to solve the problem
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise ValueError("Input is unsolvable or infeasible.")

        # Getting uav assignments
        assignments = []
        for j in uavs:
            assignments.append([])
            for i in gus:
                if solver.Value(x[i][j]):
                    assignments[j].append(i)

        ## Generating the data rates for each UAV, taking the min of path qualities and desired_throughput
        data_rates = np.zeros(self.n_tx + 1, dtype=np.int64)
        for i in range(self.n_tx):
            for j in assignments[i]:
                data_rates[i] += min(desired_throughputs[j], path_qualities[j][i])
        data_rates[-1] = np.sum(data_rates[:-1])

        return assignments, data_rates * scale


    def assignGUsWithTriObjective(self, path_coefficients, scale=1, alpha=1, beta=0.1, gamma=10, max_power=5):
        """
        Args:
            path_coefficients (np.array(num_rx, num_tx, max_num_paths)): Path quality values for all paths between each UAV-Ground User pair, can be floats
            scale (int): scales down the input data to make the computation more efficient, should be positive
            alpha (float): Optimization coefficient for the throughput maximization objective.
            beta (float): Optimization coefficient for the minimum assigned throughput objective.
            gamma (float): Optimization coefficient for the power consumption minimization objective.
            max_power (int): The maximum power to consider in the optimization, in watts
        
        Returns:
            assignments (list(list(int))): assignments[i] contains the list of GUs assigned to UAV i.
            total_throughput (int): The total theoretical maximum throughput of all UAV-GU connections.
        """
        # Creating data arrays from environment data
        capacities = np.array([x.throughput_capacity for x in self.uavs], dtype=np.int64)
        desired_throughputs = np.array([x.getDesiredThroughput() for x in self.gus], dtype=np.int64)

        model = cp_model.CpModel()
        gu_range = range(self.n_rx)
        uav_range = range(self.n_tx)
        power_range = range(1, max_power + 1)
        
        # Computing rmax array
        rmax = np.zeros((self.n_rx, self.n_tx, max_power), dtype=np.int64)
        N0 = 1 / (BOLTZMANN_CONSTANT * self.temperature)
        for i in gu_range:
            for j in uav_range:
                for k in power_range:
                    rmax[i][j][k - 1] = int(self.uavs[j].bandwidth * np.sum(np.log2(1 + N0 * k * path_coefficients[i, j, :] ** 2 / self.uavs[j].bandwidth)))
        
        if scale > 1:
            capacities = np.floor(capacities / scale).astype(np.int64)
            desired_throughputs = np.floor(desired_throughputs / scale).astype(np.int64)
            rmax = np.floor(rmax / scale).astype(np.int64)

        # 1 if GU[i] is assigned to uav[j] at level power[k] 
        z = np.array([[[model.NewBoolVar("") for k in power_range] for j in uav_range] for i in gu_range])

        # At most one assignment contraint, also enforces the power level constraint
        for i in gu_range:
            model.Add(np.sum(z[i, :, :]) <= 1)

        # Constraint for desired_throughput limit
        for j in uav_range:
            model.Add(sum(np.dot(desired_throughputs, z[:, j, k - 1]) for k in power_range) <= capacities[j])

        # Minimum throughput assignment constraints
        throughput_wastes = np.array([model.NewIntVar(0, capacities[j], "") for j in uav_range])
        for j in uav_range:
            model.Add(throughput_wastes[j] == capacities[j] - np.sum(rmax[:, j, :] * z[:, j, :]))
        
        max_throughput_waste = model.NewIntVar(0, max(capacities), "")
        for j in uav_range:
            model.Add(throughput_wastes[j] <= max_throughput_waste)

        # Adding maximizer
        coefficient = np.array([k for k in power_range])
        model.Maximize(alpha * np.sum(rmax * z) - beta * max_throughput_waste - gamma * sum(np.dot(coefficient, z[i, j, :]) for i in gu_range for j in uav_range))

        # Solving the model
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            raise ValueError("Model doesn't convege for this input")

        assignments = [[] for j in uav_range]
        for i in gu_range:
            for j in uav_range:
                for k in power_range:
                    if solver.Value(z[i, j, k - 1]):
                        assignments[j].append(i)

        throughputs = np.zeros(self.n_tx + 1, dtype=np.int64)
        for j in uav_range:
            for uav in assignments[j]:
                throughputs[j] += solver.Value(sum(rmax[uav, j, :] * z[uav, j, :]))
        throughputs[-1] = np.sum(throughputs)  # Again we can ignore 0
        
        return assignments, throughputs * scale
    

    def assignLandmarks(self, landmarks):
        """
        Assigns the UAVs current in the environment to landmarks optimially to minimize the maximum travel time
        This is an algorithm, not a heuristic, so it produces an optimal solution in O(n^3logn) time

        Args:
            landmarks (np.array(float)): The positions of the landmarks in space, shape=(num_landmarks, 3)
        
        Returns:
            Tuple(float, np.array(int)): The maximum UAV travel distance, and an array that stores the id of the landmark assigned to each UAV, shape=(num_uavs,)
        """

        n = len(self.uavs)
        m = len(landmarks)

        assert n == m
        D = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                D[i][j] = np.linalg.norm(self.uavs[i].pos - landmarks[j])
        
        flattened = np.sort(D.reshape(n * n))

        # Binary Searching with range n^2
        left = 0
        right = n * n - 1
        while left < right:
            mid = left // 2 + right // 2

            res = _perfectMatching(D <= flattened[mid])

            if res[0]:
                right = mid
            else:
                left = mid + 1

        # Swapping the matching to a standard basis
        optimal_matching = _perfectMatching(D <= flattened[left])[1]
        rtn = np.zeros(n).astype(np.int64)
        for i in range(n):
            rtn[optimal_matching[i]] = i

        return flattened[left], np.array(rtn).astype(np.int32)

    # def spawnRandomGUs(self, n_gu, z_height=1.5):
    #     """Spawn n_gu ground users at random positions, avoiding buildings."""
    #     self.gus = []  # reset GU list
    #     for i in range(n_gu):
    #         while True:
    #             # sample uniform XY in scene bounds
    #             x = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
    #             y = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
    #             z = z_height
    
    #             pos = np.array([x, y, z], dtype=np.float32)
    
    #             # check if inside building
    #             if not self.pointInBuilding(pos):
    #                 gu = GroundUser(id=i, pos=pos)  # ⚠️ you may need to simplify GroundUser class
    #                 self.gus.append(gu)
    #                 break
    # def spawnRandomGUs(self, n_gu, z_height=1.5):
    #     bbox = self.scene.mi_scene.bbox()
    #     lo = bbox.min.numpy()
    #     hi = bbox.max.numpy()
    
    #     self.gus = []
    #     for i in range(n_gu):
    #         while True:
    #             x = np.random.uniform(lo[0], hi[0])
    #             y = np.random.uniform(lo[1], hi[1])
    #             z = z_height
    #             pos = np.array([x, y, z], dtype=np.float32)
    
    #             # check not inside building
    #             if not self.pointInBuilding(pos):
    #                 gu = GroundUser(id=i, pos=pos)  # adjust constructor
    #                 self.gus.append(gu)
    #                 break

    # def spawnRandomGUs(self, n_gu, z_height=1.5, max_tries=100):
    #     """
    #     Spawn n_gu ground users at random positions inside the scene bounding box,
    #     ensuring they are not inside any building's bounding box.
    #     """
    #     bbox = self.scene.mi_scene.bbox()
    #     lo = bbox.min.numpy()
    #     hi = bbox.max.numpy()
    
    #     self.gus = []  # reset GU list if needed
    
    #     for i in range(n_gu):
    #         for _ in range(max_tries):
    #             x = np.random.uniform(lo[0], hi[0])
    #             y = np.random.uniform(lo[1], hi[1])
    #             z = z_height
    #             pos = np.array([x, y, z], dtype=np.float32)
    
    #             inside = False
    #             # Loop over scene objects (e.g., building walls/roofs)
    #             for obj in self.scene.objects.values():
    #                 try:
    #                     obj_bbox = obj.mi_obj.bbox()
    #                 except AttributeError:
    #                     continue  # some objects may not have mi_obj
    #                 bmin = obj_bbox.min.numpy()
    #                 bmax = obj_bbox.max.numpy()
    #                 if np.all(pos >= bmin) and np.all(pos <= bmax):
    #                     inside = True
    #                     break
    
    #             if not inside:
    #                 gu = GroundUser(id=i, pos=pos)  # adjust if your GroundUser has different constructor
    #                 self.gus.append(gu)
    #                 break
    #         else:
    #             raise RuntimeError(f"Could not place GU {i} outside buildings after {max_tries} tries")
    # def spawnRandomGUs(self, n_gu, z_height=1.5, max_tries=100):
    #     """
    #     Spawn n_gu ground users at random positions, avoiding buildings.
    #     Each GU is static (only 1 position in its trajectory).
    #     """
        # self.gus = []  # reset any existing GUs
        # for i in range(n_gu):
        #     for _ in range(max_tries):
        #         # sample uniform XY in scene bounds
        #         x = np.random.uniform(self.scene_bbox[0][0], self.scene_bbox[1][0])
        #         y = np.random.uniform(self.scene_bbox[0][1], self.scene_bbox[1][1])
        #         z = z_height
        #         pos = np.array([x, y, z], dtype=np.float32)
    
        #         # check it's not inside a building
        #         if not self.pointInBuilding(pos):
        #             positions = np.array([pos], dtype=np.float32)
        #             gu = GroundUser(id=i, positions=positions, height=z_height, com_type="rx")


        #             self.gus.append(gu)
        #             break
        #     else:
        #         raise RuntimeError(f"Could not place GU {i} after {max_tries} tries")
    # def spawnRandomGUs(self, n_gu, z_height=1.5, max_tries=100, mode="uniform"):
    #     self.gus = []  # clear old GUs

    #     if mode == "uniform":
    #         for i in range(n_gu):
    #             pos = sample_valid_position()
    #             self.gus.append(GroundUser(id=i, positions=np.array([pos]), height=z_height, com_type="rx"))
        
    #     elif mode == "clustered":
    #         n_clusters = max(1, n_gu // 5)
    #         centers = [sample_valid_position() for _ in range(n_clusters)]
    #         for i in range(n_gu):
    #             c = centers[i % n_clusters]
    #             pos = c + np.random.normal(scale=10, size=3)  # jitter around cluster
    #             pos[2] = z_height
    #             if not self.pointInBuilding(pos):
    #                 self.gus.append(GroundUser(id=i, positions=np.array([pos]), height=z_height, com_type="rx"))

    def spawnRandomGUs(self, n_gu, z_height=1.5, max_tries=100, mode="uniform"):
        """
        Spawn n_gu ground users at random positions, avoiding buildings.
        mode: "uniform" (default) or "clustered"
        """
        self.gus = []  # clear old GU list
    
        def sample_valid_position():
            """Try sampling until we find a position not inside a building."""
            for _ in range(max_tries):
                x = np.random.uniform(self.scene_bbox[0][0], self.scene_bbox[1][0])
                y = np.random.uniform(self.scene_bbox[0][1], self.scene_bbox[1][1])
                z = z_height
                pos = np.array([x, y, z], dtype=np.float32)
                if not self.pointInBuilding(pos):
                    return pos
            raise RuntimeError("Could not sample valid GU position after many tries")
    
        if mode == "uniform":
            for i in range(n_gu):
                pos = sample_valid_position()
                positions = np.array([pos], dtype=np.float32)
                gu = GroundUser(id=i, positions=positions, height=z_height, com_type="rx", color=np.random.rand(3))
                self.gus.append(gu)
    
        elif mode == "clustered":
            n_clusters = max(1, n_gu // 5)
            centers = [sample_valid_position() for _ in range(n_clusters)]
            for i in range(n_gu):
                c = centers[i % n_clusters]
                for _ in range(max_tries):
                    pos = c + np.random.normal(scale=10.0, size=3)
                    pos[2] = z_height
                    if not self.pointInBuilding(pos):
                        positions = np.array([pos], dtype=np.float32)
                        gu = GroundUser(id=i, positions=positions, height=z_height, com_type="rx", color=np.random.rand(3))
                        self.gus.append(gu)
                        break
        else:
            raise ValueError(f"Unsupported mode: {mode}")


    def pointInBuilding(self, pos: np.ndarray) -> bool:
        """
        Check if a 3D position is inside any building's bounding box.
        Args:
            pos: np.array([x,y,z]) position in meters
        Returns:
            True if inside a building, False otherwise
        """
        # p = mi.Point3f(pos.tolist())
        p = mi.ScalarPoint3f(pos.tolist()) 
        for name, obj in self.scene.objects.items():
            if "building" in name.lower():  # crude filter: only building meshes
                # bbox = obj.bbox()
                bbox = obj.mi_mesh.bbox()
                if bbox.contains(p):
                    return True
        return False





class BaseStation():
    def __init__(self, id, pos=np.zeros(3,), bandwidth=50, delta_t=1, signal_power=0, throughput_capacity=625000000, battery_capacity=10000):
        """
        Creates a new base station object with the specified position and communication parameters

        Args:
            id (int): The unique id of the base station
            pos (np.array(float)): The position of the base station in space, shape=(3,)
            bandwidth (float): the bandwidth of the UAV's communication system, in Mbps
            delta_t (float): The time between simulation timesteps, in seconds
            throughput_capacity (int): the throughput capacity of the UAV in bytes per second rounded to the nearest integer, default 625,000,000 bytes per second
            battery_capacity (float): the battery capacity of the UAV, in joules, defaults to 10,000 J
        """
        self.id = id
        self.pos = pos
        self.bandwidth = bandwidth
        self.delta_t = delta_t
        self.signal_power = signal_power
        self.throughput_capacity = throughput_capacity
        self.battery_capacity = battery_capacity
        self.device = None  # Initialized externally


    def computeConsumption(self):
        """
        Computes the consumption of the base station in one timestep and adds it to the base station's consumption
        """
        # self.consumption += self.signal_power * self.delta_t
        """Disabled: no consumption tracking needed."""
        return

    
    def getConsumption(self):
        """
        Gets the Base station's power consumption so far, in Joules

        Returns:
            (float): The Base Station's power consumption so far, in Joules
        """
        return self.consumption


    def lookAt(self, position=None):
        """
        Adjusts the position of the antenna to look at a specific point, default is straight down towards the ground

        Args:
            position (np.array(float)): The position to point the UAV's antenna at
        """

        if position is None:
            position = self.pos - np.array([0, 0, 1])  # A point just below the UAV
        self.device.look_at(position)

        
class UAV(BaseStation):
    def __init__(self, id, mass=1, efficiency=0.8, pos=np.zeros(3,), vel=np.zeros(3,), bandwidth=50, delta_t=1, rotor_area=0.5, signal_power=0, throughput_capacity=625000000, battery_capacity=10000):
        """
        Creates a new UAV object with the specified physical and communication parameters

        Args:
            id (int): the unique id of the UAV
            mass (float): the mass of the UAV in kg
            efficiency (float): the proportion of the UAV's power consumption that is turned into movement
            pos (np.array(3,)): the initial position of the UAV
            vel (np.array(3,)): the initial velocity of the UAV
            bandwidth (float): the bandwidth of the UAV's communication system, in Mbps
            delta_t (float): The time between simulation timesteps, in seconds
            rotor_area (float): the total area of the UAV's rotors, in square meters
            signal_power (float): the transmitter/receiver power of the UAV, in watts
            throughput_capacity (int): the throughput capacity of the UAV, in bytes per second, rounded to the nearest integer
            defaults to 625,000,000 bytes per second, or 5 Gbps
            battery_capacity (float): the battery capacity of the UAV, in joules, defaults to 10,000 J
        """
        
        # Calling super constructor for Base Station
        super().__init__(id, pos, bandwidth, delta_t, signal_power, throughput_capacity, battery_capacity)
        self.mass = mass
        self.efficiency = efficiency
        self.vel = vel
        self.rotor_area = rotor_area

        # try to fix missing attribute problem for RL training
        self.consumption = 0.0

        
    
    def p(self, t, bezier):
        """
        Computes the position of an object moving along a Bezier curve defined by bezier at time t

        Args:
            t (float): the time to compute the position at
            bezier (np.array(4, 3)): an array of bezier parameters
        """

        return (1 - t) * (1 - t) * (1 - t) * bezier[0] + 3 * t * (1 - 1) * (1 - t) * bezier[1] + 3 * t * t * (1 - t) * bezier[2] + t * t * t * bezier[3]


    def v(self, t, bezier):
        """
        Computes the velocity of an object moving along a Bezier curve defined by bezier at time t

        Args:
            t (float): the time to compute the velocity at
            bezier (np.array(4, 3)): an array of bezier parameters

        Returns:
            np.array(3,): a velocity vector
        """

        return -3 * (1 - t) * (1 - t) * bezier[0] + (9 * t * t - 12 * t + 3) * bezier[1] + (-9 * t * t + 6 * t) * bezier[2] + 3 * t * t * bezier[3]
    

    def a(self, t, bezier):
        """
        Computes the acceleration of an object moving along a Bezier curve defined by bezier at time t

        Args:
            t (float): the time to compute the acceleration at
            bezier (np.array(4, 3)): an array of bezier parameters

        Returns:
            np.array(3,): an acceleration vector
        """

        return 6 * (1 - t) * bezier[0] + (18 * t - 12) * bezier[1] + (6 - 18 * t) * bezier[2] + 6 * t * bezier[3]


    def computeConsumption(self, bezier, num_samples):
        """
        Computes the consumption from the array of cubic bezier parameters through
        numerical integration with the specified number of samples and
        includes the consumption from the signal power the UAV is currently set at

        Args:
            bezier (np.array(4, 3)): an array of bezier parameters
            num_samples (int): the number of samples used in numerical integration

        Returns:
            float: The energy consumed by the UAV, in joules
        """

        dt = self.delta_t / num_samples
        moving = 0

        for i in range(num_samples):
            moving += np.abs(np.dot(self.a(dt * i, bezier), self.v(dt * i, bezier))) * dt
        
        static = 0.5 * self.delta_t * ((self.mass * GRAVITATIONAL_ACCEL) ** 1.5) / ((self.rotor_area * AIR_DENSITY) ** 0.5)
        signal = self.delta_t * self.signal_power

        return (moving * self.mass + static) / self.efficiency + signal


    # TODO: Add object checks along the bezier curve trajectory, check for building checks and potentially UAV checks
    # The method for this could be with ray tracing, if there are no paths from UAV to outside, then it has collided with a building.
    def move(self, new_pos, new_vel, bezier_matrix):
        """
        Moves the UAV from its current position and velocity to a new position and velocity over a single time step
        Uses a cubic Bezier curve to interpolate the points to minimize the consumption
        Updates the UAV's consumption with the work done by moving

        Args:
            new_pos (np.array(3,)): the UAV's new position
            new_vel (np.array(3,)): the UAV's new velocity
        """

        # Computing the bezier
        # f = np.array([super.pos, self.vel, new_pos, new_vel])
        f = np.array([self.pos, self.vel, new_pos, new_vel])
        bezier = np.dot(bezier_matrix, f)

        # Updating consumption
        self.consumption += self.computeConsumption(bezier, NUM_INTEGRATION_SAMPLES)
        
        # Updating position and velocity
        self.pos = new_pos
        self.vel = new_vel
            

class GroundUser():
    def __init__(self, id, positions, initial_velocity=np.zeros(3,), height=1.5, bandwidth=50, com_type="rx", delta_t=1, color=np.zeros(3), desired_throughputs=np.full(150, 375000)):
        """
        Creates a new ground user with the specified parameters

        Args:
            id (int): the unique id number of the ground user
            positions (np.array(*, 3)): an array of positions over time on the xy plane
            initial_velocity (float): the velocity of the ground user at time zero
            height (float): the height of the ground user in meters
            bandwidth (float): the bandwidth of the ground user's device, in Mbps
            com_type (str): either "transmitter" or "receiver" denotes the type of the ground user
            delta_t (float): the absolute time between each time step, in seconds
            color (np.array(3,)): the color of the UAV displayed in the visualize function, expressed as RGB values in [0, 1]
            desired_throughputs (np.array(int)): the desired throughput of the ground user at each time step, in bytes per second rounded to the nearest integer
            defaults to 375000 bytes per second, or 3 Mbps
        """
        self.id = id
        self.positions = positions
        self.initial_velocity = initial_velocity
        self.time_step = 0
        self.height = height
        self.bandwidth = bandwidth
        self.delta_t = delta_t
        self.desired_throughputs = desired_throughputs
        if com_type == "tx":
            self.device = Transmitter(name=f"gu{id}", position=mi.Point3f([float(self.positions[0][0]), float(self.positions[0][1]), height]), color=color)
        elif com_type == "rx":
            self.device = Receiver(name=f"gu{id}", position=mi.Point3f([float(self.positions[0][0]), float(self.positions[0][1]), height]), color=color)
        else:
            raise ValueError("com_type must be either 'tx' or 'rx'")
    

    def update(self):
        """
        Updates the Ground User's position to the next time step and updates velocity
        """
        self.time_step += 1

        # just try to train PPO model, will remove this and use randomly generate GU positions
        if self.time_step >= len(self.positions):
            # instead of raise, loop back
            self.time_step = 0
            
        if self.time_step >= len(self.positions):
            raise ValueError(f'No more positions to update for Ground User: {self.id}')
        if self.time_step >= len(self.desired_throughputs):
            raise ValueError(f'No more desired throughputs to update for Ground User: {self.id}')


    def getCurrentPosition(self):
        """
        Gets the current position of the Ground User

        Returns:
            np.array(float): the current position of the Ground User, in m, shape=(2,)
        """
        return self.positions[self.time_step]
    

    def getTimestampPosition(self, time_step):
        """
        Gets the position of the Ground User at the specified timestep

        Args:
            step (int): The timestep to get the position of the Ground User at
        
        Returns:
            np.array(float): The current position of the Ground User, in m, shape=(2,)
        """

        # if time_step > len(self.positions) or time_step < 0:
        #     raise IndexError(f"Timestep is out of bounds for GU {self.id}")
        # else:
            # return self.positions[time_step]
        if len(self.positions) == 1:
            return self.positions[0]
        if time_step >= len(self.positions):
            return self.positions[-1]  # just stay at last
        return self.positions[time_step]

    
    def getVelocity(self):
        """
        Gets the current velocity of the Ground User

        Returns:
            float: the current velocity of the Ground User, in m/s
        """
        if self.time_step == 0:
            return self.initial_velocity
        else:
            return (self.positions[self.time_step] - self.positions[self.time_step - 1]) / self.delta_t
    

    def getDesiredThroughput(self):
        """
        Gets the desired throughput of the Ground User at the current time step

        Returns:
            float: the desired throughput of the Ground User, in bytes per second rounded to the nearest integer
        """
        return self.desired_throughputs[self.time_step]
    