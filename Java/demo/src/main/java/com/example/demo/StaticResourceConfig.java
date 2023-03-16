/*此class沒用到 , 因為前端用react , react只能讀到放在public底下的image
目前仍用mklink hard link from LPR jpg's folder to React jpg's folder */
package com.example.demo;

import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.web.servlet.WebMvcAutoConfiguration;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;


@Configuration
@EnableWebMvc
@AutoConfigureAfter(WebMvcAutoConfiguration.class)
public class StaticResourceConfig implements WebMvcConfigurer{
  @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry){
        // registry.addResourceHandler("/jpg/**").addResourceLocations("file:D:/workspace/ThaiLPR/jpg/");
        registry.addResourceHandler("/jpg/**").addResourceLocations("file:C:/Users/Cray.Hung/Desktop/timage/");
    }    


}

