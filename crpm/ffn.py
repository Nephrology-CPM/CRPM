"""Feed forward network Class.
"""

class FFN():
    """FFN class"""

    def __init__(self, desc, std=None, pre=None, post=None):
        """define model from description with options for pre and post procs
        and inital weight distribution.
        """
        from crpm.ffn_bodyplan import read_bodyplan
        from crpm.ffn_bodyplan import init_ffn

        #save weight variance parameter
        self.weightstd = std

        #get bodyplan from a file description
        if isinstance(desc, str):
            self.bodyplan = read_bodyplan(desc)

        #get bodyplan from a list description
        if isinstance(desc, list):
            self.bodyplan = desc

        #define model from bodyplan
        self.body = init_ffn(self.bodyplan, weightstd=self.weightstd)

        #link static pre-processing body
        self.pre = pre
        #append indicator in description if applicable
        if self.pre is not None:
            for layer in self.pre:
                layer["desc"] = layer["desc"] + str(' static pre-processor')

        #link static post-processing body
        self.post = post
        #append indicator in description if applicable
        if self.post is not None:
            for layer in self.post:
                layer["desc"] = layer["desc"] + str(' static post-processor')

    def reinit(self, std=None):
        """Reinitialize FFN object.

        Args:
            model: A previously created ffn model
        Returns:
            The input model with reinitialized weights and biases
        """
        import numpy as np

        #always inform user model is being reinitialized
        print("Reinitialing FFN body!")

        #reset weight distribution if given
        if std is not None:
            self.weightstd = std

        #define model from bodyplan
        self.body = init_ffn(self.bodyplan, weightstd=self.weightstd)

    def copy(self):
        """Copy FFN
        """
        from crpm.ffn_bodyplan import copy_ffn

        #init new model using current model's bodyplan
        newmodel = FFN(self.bodyplan, std=self.weightstd, pre= self.pre, post = self.post)

        #copy bodies
        newmodel.body = copy_ffn(self.body)

        #return newmodel
        return newmodel


#def ffn_body(model):
#    """ return FFN body if input is FFN object otherwise return input"""
#    #check if model is FFN object or FFN body
#    if isinstance(model, FFN):
#        return model.body
#    #else return input
#    return model

#def ffn_preprocess(model, data):
#    """ return transformed data if model is FFN object otherwise return data"""
#    from crpm.fwdprop import fwdprop
#    #check if model is FFN object with defined pre-process
#    if isinstance(model, FFN):
#        if model.pre is not None:
#            predata, _ = fwdprop(data, model.pre)
#            return predata
#    #else return input
#    return data

#def ffn_postprocess(model, data):
#    """ return transformed data if model is FFN object otherwise return data"""
#    from crpm.fwdprop import fwdprop
#    #check if model is FFN object with defined post-process
#    if isinstance(model, FFN):
#        if model.post is not None:
#            postdata, _ = data, model.post
#            return postdata
#    #else return input
#    return data
