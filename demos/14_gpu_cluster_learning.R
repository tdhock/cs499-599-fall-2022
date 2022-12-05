library(data.table)
loss.dt <- data.table(csv=Sys.glob("*.csv"))[, {
  data.table::fread(csv)
}, by=csv]

library(ggplot2)
ggplot()+
  geom_line(aes(
    epoch, loss, color=set_name),
    data=loss.dt)+
  facet_grid(model_name ~ csv, scales="free")+
  scale_y_log10()
