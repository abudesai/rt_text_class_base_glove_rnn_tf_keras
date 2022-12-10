


def __get_prep_param(pp_params,data_schema,prev_key=None): #Needs to be tested on other schemas type
    """It should handle any kind of schemas, currently tested on text dataset schema"""
    if isinstance(data_schema,(dict,list)):
        if isinstance(data_schema,dict): 
            for key in data_schema.keys():
                __get_prep_param(pp_params,data_schema[key],prev_key=key)
        else:
            for item in data_schema:
                __get_prep_param(pp_params,item,prev_key=item)
    else:
        if prev_key:
            if not prev_key in ["problemCategory","version","language","encoding"]:
                pp_params[prev_key] = data_schema
    return
    
def produce_schema_param(data_schema):
    # initiate the pp_params dict
    pp_params = {}
    __get_prep_param(pp_params,data_schema)
    return pp_params

