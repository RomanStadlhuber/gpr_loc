rm(list=ls()) #Remove all variables from workspace
library(readr)
library(latex2exp)

Y.sparse <- suppressMessages(
  as.matrix(
    read_csv("data/regression_outputs/sparse_ARD_regression/sparse_ARD_regression_regression-output__sparse_ARD_regression_regression-output_labels.csv")
  )
)
Y.full <- suppressMessages(
  as.matrix(
    read_csv("data/regression_outputs/dense_ARD_regression/dense_ARD_regression_regression-output__dense_ARD_regression_regression-output_labels.csv")
  )
)



df <- data.frame(cbind(Y.sparse[,c(2,3,4)],Y.full[,c(2,3,4)]))
colnames(df) <- c("dX","dY","dT","dX.s","dY.s","dT.s")

# linear model of D.x & Sp.X
#model.LM.X <- lm(dX ~ dX.s,data = df)
#summary(model.LM.X)

GT <- suppressMessages(as.matrix(read_csv("data/groundtruth/taurob_eval_2022-12-15-20-26-48-deltas__process_labels.csv")))

df.GT.Sparse <- data.frame(cbind(GT[,c(2,3,4)],Y.sparse[,c(2,3,4)]))
colnames(df.GT.Sparse) <- c("dX","dY","dT","dX.s","dY.s","dT.s")
# linear model of GT.x & Sp.x
lm_x_sp <- lm(dX ~ dX.s,data = df.GT.Sparse)
summary(lm_x_sp)
# GT.y ~ Sp.y
lm_y_sp <- lm(dY ~ dY.s, data = df.GT.Sparse)
summary(lm_y_sp)
# GT.theta ~ Sp.theta
lm_t_sp <- lm(dT ~ dT.s, data = df.GT.Sparse)
summary(lm_t_sp)

df.GT.Full <- data.frame(cbind(GT[,c(2,3,4)],Y.full[,c(2,3,4)]))
colnames(df.GT.Full) <- c("dX","dY","dT","dX.f","dY.f","dT.f")

# linear model of GT.x & Sp.x
lm_x_den <- lm(dX ~ dX.f,data = df.GT.Full)
summary(lm_x_den)
# GT.y ~ Sp.y
lm_y_den <- lm(dY ~ dY.f, data = df.GT.Full)
summary(lm_y_den)
# GT.theta ~ Sp.theta
lm_t_den <- lm(dT ~ dT.f, data = df.GT.Full)
summary(lm_t_den)

# set plot margin parameter
# par(mar = c(bottom, left, top, right))
par(mar = c(4.5, 4.5, 1, 1))

# X: plot the linear model comparison + regression, R^2 and legend
plot(
  GT[,2],
  Y.sparse[,2],
  pch='.',
  cex=5, 
  xlab=TeX(r'($\Delta{x}$ (GT) [m/s])'), 
  ylab=TeX(r'($\Delta{x}$ ($GP_{\mu, S}$) [m/s])')
)
abline(lm_x_sp, lwd=2, lty="longdash", col="red")
legend(
  "topleft", 
  inset=0.05, 
  legend=c("LM"),
  col=c("red"),
  lty="longdash",
  cex=0.8,
  title=TeX(sprintf(r'($R^{2} = %f$)', summary(lm_x_sp)$adj.r.squared)),
  text.font=10,
  bg='grey'
)


# Y: plot the linear model comparison + regression, R^2 and legend
plot(
  GT[,3],
  Y.sparse[,3],
  pch='.',
  cex=5, 
  xlab=TeX(r'($\Delta{y}$ (GT) [m/s])'), 
  ylab=TeX(r'($\Delta{y}$ ($GP_{\mu, S}$) [m/s])')
)
abline(lm_y_sp, lwd=2, lty="longdash", col="red")
legend(
  "topleft", 
  inset=0.05, 
  legend=c("LM"),
  col=c("red"),
  lty="longdash",
  cex=0.8,
  title=TeX(sprintf(r'($R^{2} = %f$)', summary(lm_y_sp)$adj.r.squared)),
  text.font=10,
  bg='grey'
)


# Theta: plot the linear model comparison + regression, R^2 and legend
plot(
  GT[,4],
  Y.sparse[,4],
  pch='.',
  cex=5, 
  xlab=TeX(r'($\Delta{\theta}$ (GT) [rad/s])'), 
  ylab=TeX(r'($\Delta{\theta}$ ($GP_{\mu, S}$) [rad/s])')
)
abline(lm_t_sp, lwd=2, lty="longdash", col="red")
legend(
  "topleft", 
  inset=0.05, 
  legend=c("LM"),
  col=c("red"),
  lty="longdash",
  cex=0.8,
  title=TeX(sprintf(r'($R^{2} = %f$)', summary(lm_t_sp)$adj.r.squared)),
  text.font=10,
  bg='grey'
)