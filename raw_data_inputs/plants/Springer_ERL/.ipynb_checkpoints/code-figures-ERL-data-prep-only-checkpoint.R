library(tidyverse)
library(gridExtra)
library(scales)
library(furniture)
library(stargazer)
library(ggmap)
library(dplyr)
library(viridis)
library(MatchIt)
library(reshape2)
library(plm)
library(lfe)
library(maps)
library(ggeffects)
library(mfx)
library(effects)
library(margins)
library(foreach)

# DATA PREPARATION

setwd("~/Dissertation/Project 3/bri-analysis")

# This is the data for all coal plants in Asia
data <- read.csv("coal-plant-data.csv")

# Remove plants within China
data <- filter(data, COUNTRY != "CHINA")

# Remove outliers 3 standard deviations above and below the mean emissions factor
boxplot(data$EMISFACTOR.PLATTS)
sd(data$EMISFACTOR.PLATTS, na.rm=TRUE)
subset(data, EMISFACTOR.PLATTS < 30000)
data[c(92:93, 155:162, 201:202, 1341:1342, 1913, 1996,2573:2574,2652,2705),]$EMISFACTOR.PLATTS <- 10*data[c(92:93, 155:162, 201:202, 1341:1342, 1913, 1996,2573:2574,2652,2705),]$EMISFACTOR.PLATTS
outliers <- subset(data, EMISFACTOR.PLATTS > 200000)$EMISFACTOR.PLATTS
data[which(data$EMISFACTOR.PLATTS %in% outliers),]
data <- data[-which(data$EMISFACTOR.PLATTS %in% outliers),]

# Create indicator variable = 1 if EITHER AE is Chinese or Construction is Chinese
data <- data %>% 
  mutate(AECON.CHINA = NA) %>% 
  mutate(AECON.CHINA = replace(AECON.CHINA, AE.CHINA==1 | CON.CHINA==1, 1)) %>% 
  mutate(AECON.CHINA = replace(AECON.CHINA, AE.CHINA==0 & CON.CHINA==0, 0)) %>% 
  mutate(AECON.CHINA = replace(AECON.CHINA, is.na(AE.CHINA) & CON.CHINA==0, 0)) %>%
  mutate(AECON.CHINA = replace(AECON.CHINA, AE.CHINA==0 & is.na(CON.CHINA), 0)) 

data$ANY.CHINA <- foreach(i=1:nrow(data)) %do% 
  ifelse(data$PAR.CHINA[[i]] == 1 | data$AECON.CHINA[[i]] == 1, 1, 
         ifelse(is.na(data$AECON.CHINA[[i]]) == TRUE, NA, 0))
data$ANY.CHINA <- unlist(data$ANY.CHINA)

# Create emissions intensity variable
data <- data %>% 
  mutate(EMISINT = HEATRATE.ADJ*EMISFACTOR.PLATTS*1/947800000/1000*1000)

# Calculate annual and lifetime emissions

data.nonret <- subset(data, STATUS != "RET")
data.nonret$ANNUALCO2 <- data.nonret$EMISINT*.44*8760*data.nonret$MW
data.nonret$LIFETIMECO2 <- data.nonret$EMISINT*.44*8760*data.nonret$MW*(40-(2020-data.nonret$YEAR))

sum(subset(data.nonret, ANY.CHINA == 1 & ANNUALCO2 != "NA")$ANNUALCO2)
sum(subset(data.nonret, ANY.CHINA == 0 & ANNUALCO2 != "NA")$ANNUALCO2)

sum(subset(data.nonret, ANY.CHINA == 1 & LIFETIMECO2 > 0)$LIFETIMECO2)
sum(subset(data.nonret, ANY.CHINA == 0 & LIFETIMECO2 > 0)$LIFETIMECO2)

