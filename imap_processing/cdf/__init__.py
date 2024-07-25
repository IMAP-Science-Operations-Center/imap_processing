from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes

# Load Epoch CDF attributes
cdf_manager = ImapCdfAttributes()
epoch_attrs = cdf_manager.get_variable_attributes("epoch")
