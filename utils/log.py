# logging utilities

class LogInfo(object):
  def __init__(self, log_name, verbose = 1):
    self.flog = open(log_name, 'w')
    self.verbose = verbose

  def __del__(self):
    self.mark('Finished logging.')
    self.flog.close()
  
  def mark(self, content):
    if (self.verbose == 1):
      print content
      self.flog.write(content + '\n')
      self.flog.flush()
