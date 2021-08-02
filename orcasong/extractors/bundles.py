import warnings
import numpy as np


class BundleDataExtractor:
    """ Get info present in real data. """
    def __init__(self, infile, only_downgoing_tracks=False):
        self.only_downgoing_tracks = only_downgoing_tracks

    def __call__(self, blob):
        # just take everything from event info
        if not len(blob['EventInfo']) == 1:
            warnings.warn(f"Event info has length {len(blob['EventInfo'])}, not 1")
        track = dict(zip(blob['EventInfo'].dtype.names, blob['EventInfo'][0]))
        track.update(**get_best_track(
            blob, only_downgoing_tracks=self.only_downgoing_tracks))

        track["n_hits"] = len(blob["Hits"])
        track["n_triggered_hits"] = blob["Hits"]["triggered"].sum()
        is_triggered = blob["Hits"]["triggered"].astype(bool)
        track["n_triggered_doms"] = len(np.unique(blob["Hits"]["dom_id"][is_triggered]))
        track["t_last_triggered"] = blob["Hits"]["time"][is_triggered].max()

        unique_hits = get_only_first_hit_per_pmt(blob["Hits"])
        track["n_pmts"] = len(unique_hits)
        track["n_triggered_pmts"] = unique_hits["triggered"].sum()

        if "n_hits_intime" in blob["EventInfo"]:
            n_hits_intime = blob["EventInfo"]["n_hits_intime"]
        else:
            n_hits_intime = np.nan
        track["n_hits_intime"] = n_hits_intime
        return track


def get_only_first_hit_per_pmt(hits):
    """ Keep only the first hit of each pmt. """
    idents = np.stack((hits["dom_id"], hits["channel_id"]), axis=-1)
    sorted_time_indices = np.argsort(hits["time"])
    # indices of first hit per pmt in time sorted array:
    indices = np.unique(idents[sorted_time_indices], axis=0, return_index=True)[1]
    # indices of first hit per pmt in original array:
    first_hit_indices = np.sort(sorted_time_indices[indices])
    return hits[first_hit_indices]


def get_best_track(blob, missing_value=np.nan, only_downgoing_tracks=False):
    """
    I mean first track, i.e. the one with longest chain and highest lkl/nhits.
    Can also take the best track only of those that are downgoing.
    """
    # hardcode names here since the first blob might not have Tracks
    names = ('E',
             'JCOPY_Z_M',
             'JENERGY_CHI2',
             'JENERGY_ENERGY',
             'JENERGY_MUON_RANGE_METRES',
             'JENERGY_NDF',
             'JENERGY_NOISE_LIKELIHOOD',
             'JENERGY_NUMBER_OF_HITS',
             'JGANDALF_BETA0_RAD',
             'JGANDALF_BETA1_RAD',
             'JGANDALF_CHI2',
             'JGANDALF_LAMBDA',
             'JGANDALF_NUMBER_OF_HITS',
             'JGANDALF_NUMBER_OF_ITERATIONS',
             'JSHOWERFIT_ENERGY',
             'JSTART_LENGTH_METRES',
             'JSTART_NPE_MIP',
             'JSTART_NPE_MIP_TOTAL',
             'JVETO_NPE',
             'JVETO_NUMBER_OF_HITS',
             'dir_x',
             'dir_y',
             'dir_z',
             'id',
             'idx',
             'length',
             'likelihood',
             'pos_x',
             'pos_y',
             'pos_z',
             'rec_type',
             't',
             'group_id')
    index = None
    if "Tracks" in blob:
        if only_downgoing_tracks:
            downs = np.where(blob["Tracks"].dir_z < 0)[0]
            if len(downs) != 0:
                index = downs[0]
        else:
            index = 0

    if index is not None:
        track = blob["Tracks"][index]
        return {f"jg_{name}_reco": track[name] for name in names}
    else:
        return {f"jg_{name}_reco": missing_value for name in names}


class BundleMCExtractor:
    """
    For atmospheric muon studies on mupage or corsika simulations.

    Parameters
    ----------
    inactive_du : int, optional
        Don't count mchits in this du. E.g. for ORCA4, DU 1 is inactive.
    min_n_mchits_list : tuple
        How many mchits does a muon have to produce to be counted?
        Create a seperate set of entries for each number in the tuple.
    plane_point : tuple
        For bundle diameter: XYZ coordinates of where the center of the
        plane is in which the muon positions get calculated. Should be set
        to the center of the detector!
    with_mc_index : bool
        Add a column called mc_index containing the mc run number,
        which is attempted to be read from the filename. This is for
        when the same run id/event id combination appears in mc files,
        which can happend e.g. in run by run simulations when there are
        multiplie mc runs per data run.
        Requires the filename to have a very specific format, which is
        likely not future-proof.
        TODO this would ideally not be read from the filename,
         but there is currently not other way of accessing it (07/2021).
    is_corsika : bool
        Use this when using Corsika!!!
    only_downgoing_tracks : bool
        For the best track (JG reco), consider only the ones that are downgoing.
    missing_value : float
        If a value is missing, use this value instead.

    """
    def __init__(self,
                 infile,
                 inactive_du=None,
                 min_n_mchits_list=(0, 1, 10),
                 plane_point=(17, 17, 111),
                 with_mc_index=True,
                 is_corsika=False,
                 only_downgoing_tracks=False,
                 missing_value=np.nan,
                 ):
        self.inactive_du = inactive_du
        self.min_n_mchits_list = min_n_mchits_list
        self.plane_point = plane_point
        self.with_mc_index = with_mc_index
        self.missing_value = missing_value
        self.is_corsika = is_corsika
        self.only_downgoing_tracks = only_downgoing_tracks

        self.data_extractor = BundleDataExtractor(
            infile, only_downgoing_tracks=only_downgoing_tracks)

        if self.with_mc_index:
            self.mc_index = get_mc_index(infile)
            print(f"Using mc_index {self.mc_index}")
        else:
            self.mc_index = None

    def __call__(self, blob):
        mc_info = self.data_extractor(blob)

        if self.is_corsika:
            # Corsika has a primary particle. Store infos about it
            prim_track = blob["McTracks"][0]

            # primary should be track 0 with id 0
            if prim_track["id"] != 0:
                raise ValueError("Error finding primary: mc_tracks[0]['id'] != 0")

            # direction of the primary
            mc_info["dir_x"] = prim_track.dir_x
            mc_info["dir_y"] = prim_track.dir_y
            mc_info["dir_z"] = prim_track.dir_z
            # use primary direction as plane normal
            plane_normal = np.array(prim_track[["dir_x", "dir_y", "dir_z"]].tolist())

            for fld in ("pos_x", "pos_y", "pos_z", "pdgid", "energy", "time"):
                mc_info[f"primary_{fld}"] = prim_track[fld]

            # remove primary for the following, since it's not a muon
            blob["McTracks"] = blob["McTracks"][1:]
        else:
            # In mupage, all muons in a bundle are parallel. So just take dir of first muon
            mc_info["dir_x"] = blob["McTracks"].dir_x[0]
            mc_info["dir_y"] = blob["McTracks"].dir_y[0]
            mc_info["dir_z"] = blob["McTracks"].dir_z[0]
            plane_normal = None

        # n_mc_hits of each muon in active dus
        mchits_per_muon = get_mchits_per_muon(blob, inactive_du=self.inactive_du)

        for min_n_mchits in self.min_n_mchits_list:
            if min_n_mchits == 0:
                mc_tracks_sel = blob["McTracks"]
                suffix = "sim"
            else:
                mc_tracks_sel = blob["McTracks"][mchits_per_muon >= min_n_mchits]
                suffix = f"{min_n_mchits}_mchits"

            # total number of mchits of all muons
            mc_info[f"n_mc_hits_{suffix}"] = np.sum(
                mchits_per_muon[mchits_per_muon >= min_n_mchits])
            # number of muons with at least the given number of mchits
            mc_info[f"n_muons_{suffix}"] = len(mc_tracks_sel)
            # summed up energy of all muons
            mc_info[f"energy_{suffix}"] = np.sum(mc_tracks_sel.energy)
            # bundle diameter; only makes sense for 2+ muons
            if len(mc_tracks_sel) >= 2:
                positions_plane = get_plane_positions(
                    positions=mc_tracks_sel[["pos_x", "pos_y", "pos_z"]].to_dataframe().to_numpy(),
                    directions=mc_tracks_sel[["dir_x", "dir_y", "dir_z"]].to_dataframe().to_numpy(),
                    plane_point=self.plane_point,
                    plane_normal=plane_normal,
                )
                pairwise_distances = get_pairwise_distances(positions_plane)
                mc_info[f"max_pair_dist_{suffix}"] = pairwise_distances.max()
                mc_info[f"mean_pair_dist_{suffix}"] = pairwise_distances.mean()
            else:
                mc_info[f"max_pair_dist_{suffix}"] = self.missing_value
                mc_info[f"mean_pair_dist_{suffix}"] = self.missing_value

        if self.with_mc_index:
            mc_info["mc_index"] = self.mc_index

        return mc_info


def get_plane_positions(positions, directions, plane_point, plane_normal=None):
    """
    Get the position of each muon in a 2d plane.
    Length will be preserved, i.e. 1m in 3d space is also 1m in plane space.

    Parameters
    ----------
    positions : np.array
        The position of each muon in 3d cartesian space, shape (n_muons, 3).
    directions : np.array
        The direction of each muon as a cartesian unit vector, shape (n_muons, 3).
    plane_point : np.array
        A 3d cartesian point on the plane. This will be (0, 0) in the plane
        coordinate system. Shape (3, ).
    plane_normal : np.array, optional
        A 3d cartesian vector perpendicular to the plane, shape (3, ).
        Default: Use directions if all muons are parallel, otherwise raise.

    Returns
    -------
    positions_plane : np.array
        The 2d position of each muon in the plane, shape (n_muons, 2).

    """
    if plane_normal is None:
        if not np.all(directions == directions[0]):
            raise ValueError(
                "Muon tracks are not all parallel: plane_normal has to be specified!")
        plane_normal = directions[0]

    # get the 3d points where each muon collides with the plane
    points = []
    for i in range(len(directions)):
        ndotu = np.dot(plane_normal, directions[i])
        if abs(ndotu) < 1e-6:
            raise ValueError("no intersection or line is within plane")

        w = positions[i] - plane_point
        si = -np.dot(plane_normal, w) / ndotu
        psi = w + si * directions[i] + plane_point
        points.append(psi)
    points = np.array(points)

    # Get the unit vectors of the plane. u is 0 in x, v is 0 in y.
    u = np.array([1, 0, -plane_normal[0] / plane_normal[2]])
    v = np.array([0, 1, -plane_normal[1] / plane_normal[2]])
    # norm:
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    # xy coordinates in plane
    x_dash = (points[:, 0] - plane_point[0]) / u[0]
    y_dash = (points[:, 1] - plane_point[1]) / v[1]
    position_plane = np.array([x_dash, y_dash]).T

    return position_plane


def get_pairwise_distances(positions_plane, as_matrix=False):
    """
    Get the perpendicular distance between each muon pair.

    Parameters
    ----------
    positions_plane : np.array
        The 2d position of each muon in a plane, shape (n_muons, 2).
    as_matrix : bool
        Return the whole 2D distance matrix.

    Returns
    -------
    np.array
        The distances between each pair of muons.
        1D if as_matrix is False (default), else 2D.

    """
    pos_x, pos_y = positions_plane[:, 0], positions_plane[:, 1]

    dists_x = np.expand_dims(pos_x, -2) - np.expand_dims(pos_x, -1)
    dists_y = np.expand_dims(pos_y, -2) - np.expand_dims(pos_y, -1)
    l2_dists = np.sqrt(dists_x**2 + dists_y**2)
    if as_matrix:
        return l2_dists
    else:
        return l2_dists[np.triu_indices_from(l2_dists, k=1)]


def get_mchits_per_muon(blob, inactive_du=None):
    """
    For each muon in McTracks, get the number of McHits.

    Parameters
    ----------
    blob
        The blob.
    inactive_du : int, optional
        McHits in this DU will not be counted.

    Returns
    -------
    np.array
        n_mchits, len = number of muons --> blob["McTracks"]["id"]

    """
    ids = blob["McTracks"]["id"]
    # Origin of each mchit (as int) in the active line
    origin = blob["McHits"]["origin"]
    if inactive_du:
        # only hits in active line
        origin = origin[blob["McHits"]["du"] != inactive_du]
    # get how many mchits were produced per muon in the bundle
    origin_dict = dict(zip(*np.unique(origin, return_counts=True)))
    return np.array([origin_dict.get(i, 0) for i in ids])


def get_mc_index(aanet_filename):
    # e.g. mcv5.40.mupage_10G.sirene.jterbr00005782.jorcarec.aanet.365.h5
    return int(aanet_filename.split(".")[-2])
