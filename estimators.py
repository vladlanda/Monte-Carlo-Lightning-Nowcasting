from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple,Dict,Optional
import copy
import datetime
from scipy.spatial import ConvexHull

from sys import platform
if (platform == "linux" or platform == "linux2" or platform == "win32"):
    from cuml.neighbors import KernelDensity as CUDAKernelDensity
    import cupy as cp
else:
    from sklearn.neighbors import KernelDensity


class Tracker:
    
    def __init__(self,max_tracks:int = 1000,max_dist:float = 0.2, min_samples:int = 3):
        self.max_tracks = max_tracks
        self.max_dist = max_dist
        self.min_sampels = min_samples
        self.n_updates = 0
        # print(max_dist, min_samples)
        self.dbscan = DBSCAN(eps=self.max_dist, min_samples=self.min_sampels)
        self.dbscan2 = DBSCAN(eps=self.max_dist, min_samples=1)
    
        self.tracker = {}
        self.velocities = {}


    def update(self,measurements, dt = 1.0):

        if self.n_updates <= 0: 
            db = self.dbscan.fit(measurements)
            self.n_updates +=1
        else:
            db = self.dbscan2.fit(measurements)
            self.n_updates += 1
        labels = db.labels_.astype(int)
        new_tracker = {}

        unique_labels = set(labels) - {-1}
        new_lbl_to_track = unique_labels
        for lbl in unique_labels:
            indecies = lbl == labels
            lbl_data = measurements[indecies]

            id = self.find_id(lbl_data)
            if id >= 0:
                new_tracker[id] = lbl_data
                new_lbl_to_track = new_lbl_to_track - {lbl}

        available_ids = list(set(np.arange(self.max_tracks)) - set(new_tracker.keys()))
        available_ids.sort()
        # print(new_tracker.keys(),new_lbl_to_track)

        for i,lbl in enumerate(new_lbl_to_track):
            indecies = lbl == labels
            lbl_data = measurements[indecies]

            id = available_ids[i]
            new_tracker[id] = lbl_data

        self.velocities = self.compute_velocities(new_tracker,new_lbl_to_track,dt)
        self.tracker = new_tracker

    def get_number_of_clusters(self):
        return len(self.tracker)
    
    def compute_velocities(self,new_tracker,new_ids, dt , velocity_clipping = 0.1):
        assert dt > 0
        velocities = {}

        for id,data in new_tracker.items():
            try:
                if id in new_ids: raise Exception('new id')
                _old = np.mean(self.tracker[id],axis=0)
                _new = np.mean(data,axis=0)
                velocities[id] = (_new - _old) / dt
                # Clip velocity
                # vel_norm = np.linalg.norm(velocities[id])
                # print('1',id,velocities[id],vel_norm)
                # if vel_norm > velocity_clipping and velocity_clipping > 0: 
                    # velocities[id] = velocities[id] / vel_norm * (velocity_clipping / 1000.0)
                # print('2',id,velocities[id],vel_norm)
            except:
                velocities[id] = np.array([0, 0])
        return velocities

    def find_id(self,data):
        _mean = np.mean(data,axis=0)
        _id_to_return = -1
        dist = 1e10
        for id,values in self.tracker.items():
            d = cdist([_mean],[np.mean(values,axis=0)])
            if dist > d and d <= self.max_dist:
                _id_to_return = id
        return _id_to_return

class PF:

    def __init__(self, xminmax: Tuple, yminmax: Tuple, n_particles: int = 1000):
        xmin, xmax = xminmax; assert xmin < xmax
        ymin, ymax = yminmax; assert ymin < ymax

        self.bound_range = np.array([xmax - xmin, ymax - ymin]).reshape(1, -1)
        self.bound_min = np.array([xmin, ymin]).reshape(1, -1)
        self.dimension = 4 # 2 position, 2 velocities
        self.prev_measurments = None

        self.n_particles = n_particles; assert n_particles > 0
        self.particles = np.zeros((n_particles, self.dimension))
        self.particles[:, :2] = np.random.rand(n_particles, 2) * self.bound_range + self.bound_min

        self.oring_particles = copy.deepcopy(self.particles)

        self.weights = np.ones(n_particles) / n_particles

        self.init_tracker()
    
    def init_tracker(self,max_dist:float = 0.2,min_samples:int=3):
        self.max_dist = max_dist
        self.min_samples = min_samples
        self.tracker = Tracker(max_dist=max_dist,min_samples=min_samples)

    # def predict(self, n_steps, dt=1.0, process_std_pos=1e-4, process_std_vel=5e-4):
    def predict(self, n_steps, dt=1.0, process_std_pos=1e-3, process_std_vel=1e-14):
    
        assert dt > 0
        self.particles = copy.deepcopy(self.oring_particles)
        # print(process_std_vel / (60 / dt),np.exp(-dt/60))
        for i in range(n_steps):
            noise_pos = np.random.normal(0, process_std_pos / (60 / dt), size=(self.n_particles, 2))
            noise_vel = np.random.normal(0, process_std_vel / (60 / dt), size=(self.n_particles, 2))
            self.particles[:, 0:2] += self.particles[:, 2:4] * dt + noise_pos
            self.particles[:, 2:4] += noise_vel
            self.particles = self.update_velocities(self.particles)

    # def update(self, measurements,dt = 1.0, measurement_variance=.1):

    #     dists = cdist(self.particles[:, :2], measurements)
    #     exp = np.exp(-dists**2 / measurement_variance)
    #     # exp[dists > max_radius] = 0
    #     likelihood = exp.mean(axis=1)

    #     self.weights *= likelihood + 1e-10
    #     self.weights /= np.sum(self.weights)

    #     self.tracker.update(measurements,dt)
    #     # self.resample()
    #     self.resample_clusters()
    #     self.update_velocities()
    #     self.oring_particles = copy.deepcopy(self.particles)

        # print(self.tracker.velocities)
        return self.particles

    def update_all(self,measurements, measurements_prev,dt = 1.0 ,measurement_variance = .05):

        self.particles = copy.deepcopy(self.oring_particles)

        measure = np.unique(np.concat((measurements,measurements_prev),axis=0),axis=0)
        dists = cdist(self.particles[:, :2], measure)
        exp = np.exp(-dists**1 / measurement_variance)
        # exp[dists > max_radius] = 0
        likelihood = exp.mean(axis=1)
        self.weights *= likelihood + 1e-10
        self.weights /= np.sum(self.weights)

        self.init_tracker(self.max_dist,self.min_samples)
        self.tracker.update(measure,1)
        # self.tracker.update(measurements_prev,1)
        self.tracker.update(measurements,dt)

        self.resample_clusters()
        self.particles = self.update_velocities(self.particles)
        self.oring_particles = copy.deepcopy(self.particles)

    def update_velocities(self,particles):

        if self.tracker.get_number_of_clusters() <= 0: 
            return particles
        _means = [np.mean(data,axis=0) for data in self.tracker.tracker.values()]
        dist = cdist(particles[:,:2],_means)
        dist = np.exp(-dist**2)
        weights = np.sum(dist,axis=1).reshape(-1,1) + 1e-6
        try:
            dist /= weights # n x m
        except:
            print(weights)
        est_vels = np.array(list(self.tracker.velocities.values())) 
        # print(np.ma.MaskedArray.argmin(dist,axis=1).shape)
        particles[:,2:] = dist @ est_vels

        return particles

    def resample(self):
        
        indices = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
        self.particles = self.particles[indices]

        self.weights.fill(1.0 / self.n_particles)

    def resample_clusters(self):

        
        ratio = .5
        n_clusters = self.tracker.get_number_of_clusters()
        indecies_set = set(np.arange(self.n_particles))
        for id,cluster_data in self.tracker.tracker.items():

            indecies = np.random.choice(list(indecies_set),
                                        size=int(self.n_particles * ratio / n_clusters),
                                        replace=False)
            indecies_set -= set(indecies)

            _mean = np.mean(cluster_data,axis=0)
            # _var = np.var(cluster_data,axis=0) 
            # print(_var)
            # _noise = np.random.rand(*self.particles[indecies,:2].shape) * 2 - 1
            # _noise *= self.tracker.max_dist * 2
            _noise = np.random.randn(*self.particles[indecies,:2].shape) * self.tracker.max_dist

            self.particles[indecies,:2] = _mean + _noise

        self.resample()



class TrackerV2:
    #
    MAX_SEVERE_THUNDERSTORM_SPEED_DEGh = 60/110 
    SECONDS_IN_HOUR = 60*60

    def __init__(self, 
                # xminmax: Tuple, 
                # yminmax: Tuple,
                max_dist:float = 0.2,
                min_samples:int = 3,
                max_tracks:int = 20,
                n_particles:int = 5000
                ):
        self.n_particles = n_particles
        # self.xminmax = xminmax
        # self.yminmax = yminmax
        self.max_tracks = max_tracks
        self.max_dist = max_dist
        self.min_samples = min_samples
        self.n_updates = 0
        # print(max_dist, min_samples)
        self.dbscan = DBSCAN(eps=self.max_dist, min_samples=self.min_samples)
        self.particle_filters:Dict[str,ParticleFilterV2] = {}
        # self.velocities:Dict[str,np.array] = {}
        # self.velocities:Dict[str,Tuple[np.array,np.array]] = {}
        self.velocities = np.zeros((0,4))

        self.seconds2minutes = 1./60.0
    
    def get_all_particles(self):

        if len(self.particle_filters) <= 0:
            return np.zeros((0,4))
        particles = list(self.particle_filters.values())
        particles = [p.particles for p in particles]
        return np.concat(particles,axis=0)

    def init_tracker(self,max_dist,min_samples):
        self.max_dist = max_dist
        self.min_samples = min_samples
        self.dbscan = DBSCAN(eps=self.max_dist, min_samples=self.min_samples)
        self.particle_filters = {}
        self.clusters = {}
        # self.velocities:Dict[str,np.array] = {}
        # self.velocities:Dict[str,Tuple[np.array,np.array]] = {}

        # self.particle_filters[0] = ParticleFilterV2(self.xminmax,
        #                                             self.yminmax,
        #                                             n_particles=self.n_particles)
        # self.velocities[0] = np.array([0,0]).reshape(1,-1)

    def update_all(self,measurements, measurements_prev, dt: datetime.timedelta):
        unique_measure = np.concat((measurements,measurements_prev),axis=0)
        dbscan = self.dbscan.fit(unique_measure)
        lbls_set = set(dbscan.labels_) - set({-1})
        self.velocities = np.zeros((0,4))

        max_area = -1
        for lbl in lbls_set:
            lbl_indx = np.where(dbscan.labels_ == lbl)[0]
            cls = unique_measure[lbl_indx,:]
            ''' Split cls (class of both) into curr and prev'''
            # Check if it in curr
            curr_idx = np.isin(measurements,cls).all(axis=1)
            prev_idx = np.isin(measurements_prev,cls).all(axis=1)
            curr = measurements[curr_idx,:]
            prev = measurements_prev[prev_idx,:]
            # print()
            if curr_idx.sum() < 3 or prev_idx.sum() <= 0 : 
                continue
            try:
                hull  = ConvexHull(curr)
            except: continue
            if hull.area > max_area:
                max_area = hull.area

        for lbl in lbls_set:
            lbl_indx = np.where(dbscan.labels_ == lbl)[0]
            cls = unique_measure[lbl_indx,:]
            ''' Split cls (class of both) into curr and prev'''
            # Check if it in curr
            curr_idx = np.isin(measurements,cls).all(axis=1)
            prev_idx = np.isin(measurements_prev,cls).all(axis=1)
            curr = measurements[curr_idx,:]
            prev = measurements_prev[prev_idx,:]
            # print()
            if curr_idx.sum() < 3 or prev_idx.sum() <= 0 : 
                continue
            try:
                hull  = ConvexHull(curr)
            except: continue
            xminmax = (np.min(curr[:,0]),np.max(curr[:,0]))
            yminmax = (np.min(curr[:,1]),np.max(curr[:,1]))

            pf = ParticleFilterV2(xminmax,
                                  yminmax,
                                  n_particles=self.n_particles)

            velocity = np.array([0.,0.])
            if prev_idx.sum() > 0:
                velocity = curr.mean(axis=0) - prev.mean(axis=0)

            velocity /= (self.seconds2minutes * dt.seconds)
            if np.linalg.norm(velocity) > self.MAX_SEVERE_THUNDERSTORM_SPEED_DEGh:
                print(velocity,np.linalg.norm(velocity))

            pf.update_velocity(velocity)
            pf.update(hull,max_area)

            self.particle_filters[lbl] = pf

            pos_vel = np.array([*np.mean(curr,axis=0),*velocity]).reshape(1,-1)
            self.velocities = np.concat((self.velocities,pos_vel),axis=0)
        
    #     self.interpolate_velocities(self.velocities)

    # def interpolate_velocities(self,velocities:np.array):
    #     if velocities.shape[0] <= 0: return
    #     loc,vel = velocities[:,:2],velocities[:,2:]

    #     for lbl,pf in self.particle_filters.items():
    #         dists = cdist(pf.particles[:,:2],loc)
    #         nume = np.exp(-dists)
    #         deno = nume.sum(axis=1,keepdims=True)
    #         softmax = nume/deno
    #         particles_velocities = softmax @ vel
    #         pf.update_velocities(particles_velocities)
    #     print(softmax[0])


    def predict(self,
                n_steps,
                dt:datetime.timedelta,
                pred_dt:datetime.timedelta = datetime.timedelta(minutes=60)):
        
        dt_ratio = pred_dt.seconds**2 / dt.seconds * self.seconds2minutes
        # print('dt_ratio',dt_ratio,pred_dt,dt)
        for lbl,pf in self.particle_filters.items():
            pf.predict(n_steps,dt_ratio,self.velocities)
            # pf.predict(n_steps,dt_ratio)

    def get_gaussian_estimation(self,grid_coords,resolution_x,resolution_y,thr = 0,bandwith=0.1):
        assert thr <= 1 and thr >= 0
        # Z_SUM    = np.zeros((resolution_y,resolution_x))
        if len(self.particle_filters) <= 0: return np.zeros((resolution_y,resolution_x))
        # factor = 1. / len(self.particle_filters)
        CUDA = False
        if platform == "linux" or platform == "linux2" or platform == "win32":
            CUDA = cp.cuda.is_available()
            # CUDA = False
        if CUDA:
            kde = CUDAKernelDensity(bandwidth=bandwith, kernel='gaussian')
        else:
            kde = KernelDensity(bandwidth=bandwith, kernel='gaussian')


        XYs = np.zeros((0,2))

        Z = np.zeros((resolution_y,resolution_x))


        for lbl,pf in self.particle_filters.items():
            xy = pf.particles[:,:2]
            XYs = np.concat((XYs,xy),axis=0)
            kde.fit(xy)
            # print(type(grid_coords),CUDA,type(kde))
            z = cp.exp(kde.score_samples(cp.array(grid_coords))).reshape(resolution_y,resolution_x).get() if CUDA else \
                np.exp(kde.score_samples(grid_coords)).reshape(resolution_y,resolution_x)
            # z = cp.exp(kde.score_samples(grid_coords)).get() if CUDA else \
            #     np.exp(kde.score_samples(grid_coords))
            try:
                # z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z /= np.max(z)
            except:
                # z is filled with 0
                z = np.zeros((resolution_y,resolution_x))
                pass
            # print(np.min(z),np.max(z))
            Z = np.maximum(Z,z)
            # Z_FACTOR[z > 0.01] += 1
            # Z_SUM += z

        # Z_FACTOR[Z_FACTOR <= 0] = 1
        # Z_FACTOR = 1.0 / Z_FACTOR
        # Z = Z_FACTOR * Z_SUM

        # kde.fit(XYs)
        # Z = cp.exp(kde.score_samples(grid_coords)).reshape(resolution_y,resolution_x).get() if CUDA else \
        #     np.exp(kde.score_samples(grid_coords)).reshape(resolution_y,resolution_x)
        # Z /= np.sum(Z)
        if thr > 0 and thr < 1.:
            # if thr >= 1: thr=.99
            dens = Z.reshape(-1)
            sorted_dens = np.sort(dens)[::-1]
            cumulative_sum = np.cumsum(sorted_dens)
            total_sum = np.sum(dens)
            cumulative_prob = cumulative_sum / total_sum
            threshold_index = np.where(cumulative_prob >= thr)[0][0]
            threshold_value = sorted_dens[threshold_index]
            Z[Z < threshold_value] = 0
            Z[Z >= threshold_value] = 1



        # if thr > 0:
        #     Z[Z >= thr] = 1.
        #     Z[Z < thr ] = 0.

        
        # print(np.min(Z),np.max(Z))

        return Z
      
class ParticleFilterV2:
    
    def __init__(self, xminmax: Tuple, yminmax: Tuple, n_particles: int = 1000,pos_noise = 5e-2,vel_noise=3e-4):
        xmin, xmax = xminmax; assert xmin < xmax
        ymin, ymax = yminmax; assert ymin < ymax

        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.bound_range = np.array([xmax - xmin, ymax - ymin]).reshape(1, -1)
        self.bound_min = np.array([xmin, ymin]).reshape(1, -1)
        self.dimension = 4 # 2 position, 2 velocities
        self.prev_measurments = None
        self.noise_factor = 1

        self.n_particles = n_particles; assert n_particles > 0
        self.particles = np.zeros((n_particles, self.dimension))
        self.particles[:, :2] = np.random.rand(n_particles, 2) * self.bound_range + self.bound_min

        self.oring_particles = copy.deepcopy(self.particles)

        self.weights = np.ones(n_particles) / n_particles

    def predict(self, n_steps, dt:float, velocities:Optional[np.array] = None):
    
        assert dt > 0
        self.particles = copy.deepcopy(self.oring_particles)
        for i in range(n_steps):
            if not velocities is None:
                self.particles = self.update_velocities(velocities)
            noise_pos = np.random.normal(0, self.pos_noise * self.noise_factor, size=(self.n_particles, 2))
            noise_vel = np.random.normal(0, self.vel_noise, size=(self.n_particles, 2))
            self.particles[:, 0:2] += self.particles[:, 2:4] * dt + noise_pos
            # self.particles[:, 2:4] += noise_vel
            # self.particles = self.update_velocities(self.particles)


        return self.particles
    
    def update(self,hull: ConvexHull,max_area: float = -1,measurement_variance:float = 0.1):
        self.noise_factor = hull.area
        self.particles = copy.deepcopy(self.oring_particles)

        particle_proportions = hull.area / max_area
        if max_area <= 1e-16:
            particle_proportions = 1

        
        # n_particles = int(particle_proportions * self.n_particles)
        # proportional_indecies = np.random.choice(self.n_particles,n_particles)
        # self.n_particles = n_particles
        # self.particles = self.particles[proportional_indecies,:]
        # self.weights = np.ones(n_particles) / n_particles
        

        # self.particles = self.particles[np.random.choice(self.n_particles,n_particles),:]
        # print()

        distances,mask = self.estimate_distance_and_mask(hull,self.particles)
        max_dist = np.max(distances,axis=1)
        weights = np.exp(-(max_dist)**2 / measurement_variance)    
        
        self.weights[mask] = self.n_particles * np.max(max_dist) * 2
        self.weights[~mask] = weights[~mask]
        self.weights /= np.sum(self.weights)

        self.resample()
        
        self.pertubate(self.pos_noise * self.noise_factor,self.vel_noise)
        # self.pertubate(self.pos_noise,self.vel_noise)
        self.oring_particles = copy.deepcopy(self.particles)

    def pertubate(self,pos_noise=None,vel_noise=None):
        if not pos_noise: pos_noise = self.pos_noise * self.noise_factor
        if not vel_noise: vel_noise = self.vel_noise
        noise_pos = np.random.normal(0,pos_noise,size=(self.n_particles,2))
        noise_vel = np.random.normal(0,vel_noise,size=(self.n_particles,2))
        # noise_pos = np.random.uniform(-self.pos_noise,self.pos_noise,size=(self.n_particles,2))
        # noise_vel = np.random.uniform(-self.vel_noise,self.vel_noise,size=(self.n_particles,2))
        noise = np.concat((noise_pos,noise_vel),axis=1)
        self.particles += noise

    def resample(self):
        indices = np.random.choice(self.n_particles, self.n_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.n_particles)
        
    def estimate_distance_and_mask(self,hull,particles,tolerance=1e-12):
        if particles.shape[-1] > 2:
            particles = particles[:,:2] # use only positions
        
        particles_mat = np.concat((particles,np.ones((particles.shape[0],1))),axis=1)
        distances = particles_mat @ hull.equations.T
        mask = (distances <= tolerance).all(axis=1)
        
        return distances,mask
    
    def update_velocity(self,velocity):
        '''
            Velocity [dx/dt,dy/dt]
        '''
        velocity = velocity.reshape(1,-1) 
        self.oring_particles[:,2:] = velocity

    def update_velocities(self,velocities:np.array):
        if velocities.shape[0] <= 0: return
        loc,vel = velocities[:,:2],velocities[:,2:]
        dists = cdist(self.particles[:,:2],loc)
        nume = np.exp(-dists)
        deno = nume.sum(axis=1,keepdims=True)
        softmax = nume/deno

        ''' Velocity weighting'''
        norm_vel =  np.linalg.norm(vel,axis=1,keepdims=True)
        norm_vel[norm_vel < 1e-12] = 1e-12  # Avoid division by zero
        self.particles[:,2:] = (softmax @ (vel / norm_vel)) * (softmax @ norm_vel)

        ''' Alternative way to update velocities'''
        # self.particles[:,2:] = softmax @ vel

        return self.particles


def testing():

    ildn_file = 'ILDN 26-31_01_24.xlsx'
    df = pd.read_excel(ildn_file,index_col=False,
                            sheet_name='Sheet1')

    df = df.sort_values('UTC',ignore_index=True)

    settings = Settings()
    tracker = TrackerV2()

    curr_time = datetime.datetime(2024,1,30,10,20,44)
    curr_measure_mask = (df['UTC'] <= curr_time)&(df['UTC'] > curr_time - settings.history_window)
    curr_measure = df[curr_measure_mask]

    prev_measure_mask = (df['UTC'] <= curr_time - settings.dt)&(df['UTC'] > curr_time - settings.dt - settings.history_window)
    prev_measure = df[prev_measure_mask]

    np_curr_measure = curr_measure[['lon','lat']].to_numpy()
    np_prev_measure = prev_measure[['lon','lat']].to_numpy()
    
    tracker.init_tracker(settings.max_dist,settings.min_samples)
    tracker.update_all(np_curr_measure,np_prev_measure,settings.dt)

    expand  = settings.expand

    xminmax = (settings.nswe['w']-expand, settings.nswe['e']+expand)
    yminmax = (settings.nswe['s']-expand, settings.nswe['n']+expand)

    print(settings.contour_resolution_x,settings.contour_resolution_y)
    xlin = np.linspace(xminmax[0], xminmax[1],settings.contour_resolution_x)
    ylin = np.linspace(yminmax[0], yminmax[1],settings.contour_resolution_y)
    X,Y = np.meshgrid(xlin, ylin)



    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io.img_tiles import GoogleTiles

    bbox = [31, 37, 31, 34]
    google_tiles = GoogleTiles(style='satellite')

    fig = plt.figure(figsize=(17, 6))
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.set_extent(bbox)

    ax = fig.add_subplot(1, 2, 1, projection=google_tiles.crs)
    ax.set_extent(bbox,crs=ccrs.PlateCarree())



    lon_range = bbox[1] - bbox[0]
    if lon_range < 1:
        zoom_level = 11
    elif lon_range < 3:
        zoom_level = 10
    elif lon_range < 5:
        zoom_level = 9
    elif lon_range < 10:
        zoom_level = 8
    else:
        zoom_level = 7

    # Step 3: Add the Google Tiles imagery to the map
    ax.add_image(google_tiles, zoom_level)

    grid_coords = np.vstack([X.ravel(), Y.ravel()]).T
    Z = tracker.get_gaussian_estimation(grid_coords,
                                        settings.contour_resolution_x, 
                                        settings.contour_resolution_y,
                                        thr=.99
                                        )
    
    # z_shape = Z.shape
    # dens = Z.reshape(-1)
    # sorted_dens = np.sort(dens)[::-1]
    # cumulative_sum = np.cumsum(sorted_dens)
    # total_sum = np.sum(dens)
    # cumulative_prob = cumulative_sum / total_sum
    # threshold_index = np.where(cumulative_prob >= 0.99)[0][0]
    # threshold_value = sorted_dens[threshold_index]
    # Z[Z < threshold_value] = 0
    # Z = np.ma.masked_array(Z, Z < 0.01)
    # Z = sorted_dens.reshape(z_shape)
    # print(sorted_dens)
    
    cs = ax.contourf(X, Y, Z, levels=25, cmap='plasma',alpha=.7,zorder=4,transform=ccrs.PlateCarree())
    # cax = plt.axes([0.95, 0.05, 0.05,0.9])
    cbar = fig.colorbar(cs,fraction=0.028, pad=0.04)
    cbar.set_label("Normalized density")

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=.05)
    # fig.colorbar(cs, cax=cax)

    # ax.scatter(*np_prev_measure.T,alpha=.6,label='Previous',s=30,edgecolors='none',facecolors='red',zorder=3,transform=ccrs.PlateCarree())
    # ax.scatter(*np_curr_measure.T,alpha=1,label='Current',s=120,edgecolors='white',facecolors='none',zorder=3,transform=ccrs.PlateCarree())
    ax.scatter(*np_curr_measure.T,alpha=1,label='Current',s=25,color='red',zorder=5,transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)#, linewidth=0.8, edgecolor='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=3)
    # ax.add_feature(cfeature.RIVERS, edgecolor='cyan', zorder=3)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='black', alpha=0.5, zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    ax.text(0.5, -0.10, 'Longitude', transform=ax.transAxes, ha='center', fontsize=12)
    ax.text(-0.11, 0.5, 'Latitude', transform=ax.transAxes, va='center', rotation='vertical', fontsize=12)
    # ax.set_title(f'"Current" and "Previous" measurments with \n60 minutes window and 30 minutes delta time.', fontsize=18, fontweight='bold')
    ax.set_title(f'Lightning strikes density estimation from \nILDN at {curr_time.strftime("%d-%m-%Y %H:%M:%S")} UTC', fontsize=18, fontweight='bold')
    legend = plt.legend(title='Measurements',loc='upper right',labelcolor='black')
    legend.get_frame().set_facecolor('gray')



    # 3d plot
    ax = fig.add_subplot(1,2,2,projection = '3d')
    surf = ax.plot_surface(X, Y, Z, cmap='inferno',
                       linewidth=0, antialiased=False)


    plt.show()


if __name__ == '__main__':

    import pandas as pd
    from applicationV2 import Settings
    from estimators import TrackerV2
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    testing()
