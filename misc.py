import pandas as pd
import numpy as np
import h5py

# miscellaneous utility functions
def pandas_to_hdf(group, pandas_object, dset_name, **kwargs):
    """ Store a pandas object to a group of an HDF file

    Parameters
    ----------
    **kwargs : dict
        other parameters to h5py.group.create_dataset

    Returns
    -------
    None
    """
    # create new dataset within `group`
    dset = group.require_dataset(name=dset_name, shape=pandas_object.shape,
        dtype=pandas_object.values.dtype)
    dset[...] = pandas_object.values

    # Need to store all attributes of `pandas_object`;
    # index is stored irrespective of whether it is Series, DataFrame or Panel

    # other attributes are stored conditional on `pandas_object` being
    # Series, DataFrame or Panel
    if pandas_object.ndim == 1:
        # Series: index only (convert to string and encode)
        dset.attrs["index"] = pandas_object.index.map(
            lambda x: x.strftime("%Y%m%d")).astype('S')

    if pandas_object.ndim == 2:
        # DataFrame: index + columns (encode all strings if ASCII)
        dset.attrs["index"] = pandas_object.index.map(
            lambda x: x.strftime("%Y%m%d")).astype('S')
        col = [p.encode() if isinstance(p, str) else p \
            for p in pandas_object.columns]
        dset.attrs["columns"] = col

    elif pandas_object.ndim == 3:
        # Panel: index + two axes (NB: "items" are stored as "index")
        # (encode all strings if ASCII)
        dset.attrs["index"] = pandas_object.items.map(
            lambda x: x.strftime("%Y%m%d")).astype('S')
        maj_ax = [p.encode() if isinstance(p, str) else p \
            for p in pandas_object.major_axis]
        min_ax = [p.encode() if isinstance(p, str) else p \
            for p in pandas_object.minor_axis]
        dset.attrs["major_axis"] = maj_ax
        dset.attrs["minor_axis"] = min_ax


def hdf_to_pandas(dset):
    """ Construct pandas object from HDF dataset.
    Assuming that the dataset is structured correctly, uses data and
    attributes to retrieve a pandas object. The object returned and the
    attributes required depend upon the number of dimensions of dset[:] as
    follows:
    1D: returns Series, requires "index"
    2D: returns DataFrame, requires "index" and "columns"
    3D: returns Panel, requires "index", "major_axis" and "minor_axis"

    Parameters
    ----------
    dset : h5py.dataset
        with data and stored as attributes

    Returns
    -------
    res : pandas.Series/DataFrame/Panel
        with data = dset[:]
    """
    # dset = hangar["covariances"]
    # collect data as numpy.ndarray
    data = dset[:]
    # fetch index from attributes
    idx = pd.to_datetime(dset.attrs["index"], format="%Y%m%d")
    # determine if Series, DataFrame or Panel (looking at dimensions)
    if data.ndim == 1:
        # Series: index only
        res = pd.Series(data=data,
        index=idx)
    elif data.ndim == 2:
        # DataFrame: index and columns (decode column names if not strings)
        col = [p.decode() if isinstance(p, bytes) else p \
            for p in dset.attrs["columns"]]

        res = pd.DataFrame(data=data,
            index=idx,
            columns=col)

    elif data.ndim == 3:
        # Panel: items (=index) and two axes
        # decode strings if needed
        maj_ax = [p.decode() if isinstance(p, bytes) else p \
            for p in dset.attrs["major_axis"]]
        min_ax = [p.decode() if isinstance(p, bytes) else p \
            for p in dset.attrs["minor_axis"]]

        res = pd.Panel(data=data,
            items=idx,
            major_axis=maj_ax,
            minor_axis=min_ax)

    return res

def panel_rolling(data, fun, **kwargs):
    """ Implementation of .rolling(`**kwargs`).apply(`fun`) for Panel

    Note that in order to replicate skipna=True behaviour of pandas built-ins
    like mean(), sum() etc. one needs to provide `fun` which can skip nans.
    Use "fun=np.nanmean" and the like to get the desired behaviour.

    Parameters
    ----------
    **kwargs : dict
        of parameters to .rolling method
    """
    # data = cormat_panel
    # kwargs = {"window" : 5}
    # fun = np.nanmean
    res = data.copy()
    # iterate over cross-sectional (with same time index for all) slices
    # NB: data.loc[:,a,:].columns are the items axis of Panel, so the whole
    # thing is transposed which is why one needs "axis=1" down there
    for label in data.major_axis:
        res.loc[:,label,:] = \
            res.loc[:,label,:].rolling(axis=1, **kwargs).apply(fun)

    return res
